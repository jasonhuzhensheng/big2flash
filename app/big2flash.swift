// big2flash: menu-bar app that runs the local MLX engine (engine/server.py) and opens Claude Code on it.
import AppKit
import SwiftUI

let home = FileManager.default.homeDirectoryForCurrentUser
let support = home.appendingPathComponent("Library/Application Support/big2flash")
let python = support.appendingPathComponent("venv/bin/python")
let engineDir = Bundle.main.resourceURL!.appendingPathComponent("engine")
let claudeLocal = engineDir.appendingPathComponent("claude-local.sh")

func shq(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

struct Model: Identifiable, Hashable {
    let name: String, path: String, size: Int64
    var id: String { path }
    var mtp: Bool { FileManager.default.fileExists(atPath: path + "/mtp/model.safetensors") }
    var label: String {
        (size > 0 ? "\(name)  (\(String(format: "%.0f", Double(size) / 1e9)) GB)" : name) + (mtp ? " · MTP" : "")
    }
}

// An MLX checkpoint mlx-lm can load: a folder with config.json carrying a quantization (or plain
// safetensors weights) - 模型库 lists some of these as "safetensors" rather than "mlx".
func isMLXFolder(_ path: String) -> Bool {
    guard let d = FileManager.default.contents(atPath: path + "/config.json"),
          let c = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return false }
    return c["quantization"] != nil || (c["text_config"] as? [String: Any])?["quantization"] != nil
}

enum Phase: Equatable {
    case stopped, settingUp, loading, ready, failed(String)
    var text: String {
        switch self {
        case .stopped: return "已停止"
        case .settingUp: return "正在安装运行环境…"
        case .loading: return "正在加载模型…"
        case .ready: return "运行中"
        case .failed(let why): return why
        }
    }
    var busy: Bool { self == .settingUp || self == .loading || self == .ready }
}

@MainActor final class Engine: ObservableObject {
    static let shared = Engine()
    @Published var phase = Phase.stopped
    @Published var models: [Model] = []
    @Published var log: [String] = []
    @Published var lastRequest = ""
    @Published var stats = EngineStats()
    @Published var contextLimit = 0  // what the running engine actually allows (it lowers the setting to fit memory)
    @Published var modelPath = UserDefaults.standard.string(forKey: "model") ?? "" {
        didSet { UserDefaults.standard.set(modelPath, forKey: "model") }
    }
    @Published var port = UserDefaults.standard.object(forKey: "port") as? Int ?? 8080 {
        didSet { UserDefaults.standard.set(port, forKey: "port") }
    }
    @Published var thinking = UserDefaults.standard.string(forKey: "thinking") ?? "auto" {
        didSet { UserDefaults.standard.set(thinking, forKey: "thinking") }
    }
    @Published var mtp = UserDefaults.standard.object(forKey: "mtp") as? Bool ?? true {
        didSet { UserDefaults.standard.set(mtp, forKey: "mtp") }
    }
    @Published var maxContext = UserDefaults.standard.object(forKey: "maxContext") as? Int ?? 98304 {
        didSet { UserDefaults.standard.set(maxContext, forKey: "maxContext") }
    }
    private var proc: Process?
    private var stopping = false
    private var partial = ""

    var url: String { "http://127.0.0.1:\(port)" }
    var hasRuntime: Bool { FileManager.default.isExecutableFile(atPath: python.path) }
    var terminalCommand: String {
        "BIG2FLASH_URL=\(url) BIG2FLASH_COMPACT_WINDOW=\(contextLimit > 0 ? contextLimit : maxContext) sh \(shq(claudeLocal.path))"
    }

    init() { refreshModels() }

    // Chat models in MLX format, from the 模型库 app (modelshelf); the chosen folder is kept even if unlisted.
    func refreshModels() {
        Task.detached {
            var found: [Model] = []
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            p.arguments = [home.appendingPathComponent(".local/bin/modelshelf").path, "find", "", "--json"]
            let out = Pipe()
            p.standardOutput = out
            p.standardError = FileHandle.nullDevice
            if (try? p.run()) != nil {
                let data = out.fileHandleForReading.readDataToEndOfFile()
                p.waitUntilExit()
                if let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    for r in rows where r["kind"] as? String == "chat" {
                        guard let name = r["name"] as? String, let path = r["path"] as? String,
                              r["format"] as? String == "mlx" || (r["format"] as? String == "safetensors" && isMLXFolder(path))
                        else { continue }
                        found.append(Model(name: name, path: path, size: (r["size"] as? NSNumber)?.int64Value ?? 0))
                    }
                }
            }
            // models copied next to the app (internal SSD: unaffected by the external drive)
            let local = support.appendingPathComponent("models")
            for dir in (try? FileManager.default.contentsOfDirectory(atPath: local.path)) ?? [] {
                let path = local.appendingPathComponent(dir).path
                guard isMLXFolder(path) else { continue }
                let files = (try? FileManager.default.subpathsOfDirectory(atPath: path)) ?? []
                let size = files.reduce(Int64(0)) { sum, f in
                    sum + ((try? FileManager.default.attributesOfItem(atPath: path + "/" + f)[.size] as? Int64) ?? 0)
                }
                found.append(Model(name: dir + "（内置硬盘）", path: path, size: size))
            }
            let list = found.sorted { $0.size < $1.size }
            await MainActor.run {
                self.models = list
                if self.modelPath.isEmpty, let first = list.first(where: { $0.mtp && $0.path.hasPrefix(support.path) })
                    ?? list.first(where: { $0.name.contains("27B") }) ?? list.first {
                    self.modelPath = first.path
                }
            }
        }
    }

    var modelChoices: [Model] {
        if modelPath.isEmpty || models.contains(where: { $0.path == modelPath }) { return models }
        return models + [Model(name: (modelPath as NSString).lastPathComponent, path: modelPath, size: 0)]
    }

    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = "选择一个 MLX 模型文件夹（含 config.json 和 .safetensors）"
        NSApp.activate(ignoringOtherApps: true)
        if panel.runModal() == .OK, let u = panel.url { modelPath = u.path }
    }

    func start() {
        guard proc == nil else { return }
        guard hasRuntime else { phase = .failed("缺少运行环境"); return }
        guard FileManager.default.fileExists(atPath: modelPath + "/config.json") else {
            phase = .failed("找不到模型（外接盘没插？）"); return
        }
        log.removeAll()
        lastRequest = ""
        contextLimit = 0
        run(python, ["-u", engineDir.appendingPathComponent("server.py").path, "--model", modelPath,
                     "--port", String(port), "--thinking", thinking, "--max-context", String(maxContext),
                     "--mtp", mtp ? "1" : "0"],
            phase: .loading)
    }

    // Recreates the Python environment the engine runs in (venv in Application Support, pinned packages).
    func setUp() {
        guard proc == nil else { return }
        let uv = [home.appendingPathComponent(".local/bin/uv").path, "/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
        guard let uv else { phase = .failed("没找到 uv（brew install uv）"); return }
        let req = Bundle.main.resourceURL!.appendingPathComponent("requirements.txt").path
        let venv = support.appendingPathComponent("venv").path
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        log.removeAll()
        run(URL(fileURLWithPath: "/bin/sh"),
            ["-c", "\(shq(uv)) venv --python 3.12 \(shq(venv)) && \(shq(uv)) pip install --python \(shq(venv + "/bin/python")) -r \(shq(req))"],
            phase: .settingUp)
    }

    private func run(_ exe: URL, _ args: [String], phase next: Phase) {
        let p = Process()
        p.executableURL = exe
        p.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil; return }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in Engine.shared.ingest(text) }
        }
        p.terminationHandler = { p in
            let code = p.terminationStatus
            Task { @MainActor in Engine.shared.exited(code) }
        }
        do {
            try p.run()
        } catch {
            phase = .failed("启动失败：\(error.localizedDescription)")
            return
        }
        proc = p
        stopping = false
        phase = next
    }

    private func ingest(_ text: String) {
        partial += text
        var lines = partial.components(separatedBy: "\n")
        partial = lines.removeLast()
        for line in lines {
            // tqdm-style progress lines rewrite themselves with \r: keep the last state only
            let clean = line.components(separatedBy: "\r").last { !$0.isEmpty } ?? ""
            if clean.isEmpty { continue }
            log.append(clean)
            if clean.hasPrefix("listening on") { phase = .ready }
            if clean.hasPrefix("context limit "), let n = Int(clean.split(separator: " ")[2]) { contextLimit = n }
            if clean.hasPrefix("["), clean.contains("prompt") {
                lastRequest = clean
                stats.add(logLine: clean)
            }
        }
        if log.count > 3000 { log.removeFirst(log.count - 3000) }
    }

    private func exited(_ code: Int32) {
        let wasSetup = phase == .settingUp
        proc = nil
        if stopping { phase = .stopped }
        else if wasSetup { phase = code == 0 ? .stopped : .failed("安装失败，见日志") }
        else {
            let inUse = log.contains { $0.contains("Address already in use") }
            phase = .failed(inUse ? "端口 \(port) 已被占用" : "引擎退出（代码 \(code)），见日志")
        }
    }

    func stop() {
        guard let p = proc else { return }
        stopping = true
        p.terminate()
        let pid = p.processIdentifier
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            if p.isRunning { kill(pid, SIGKILL) }
        }
    }

    func stopForQuit() {
        guard let p = proc else { return }
        stopping = true
        p.terminate()
        let deadline = Date().addingTimeInterval(3)
        while p.isRunning && Date() < deadline { usleep(50_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
    }

    // Opens Terminal in the chosen project with Claude Code pointed at this engine.
    func openClaude() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = "选择要让 Claude Code 工作的项目文件夹"
        panel.prompt = "打开"
        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        let dir = support.appendingPathComponent("launch")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("claude-\(Int(Date().timeIntervalSince1970)).command")
        let script = "#!/bin/zsh -l\nexport PATH=\"$HOME/.local/bin:/opt/homebrew/bin:$PATH\"\ncd \(shq(folder.path)) || exit 1\nrm -f \(shq(file.path))\nexec \(terminalCommand)\n"
        do {
            try script.write(to: file, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
        } catch {
            phase = .failed("写启动脚本失败：\(error.localizedDescription)"); return
        }
        let terminal = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal")!
        NSWorkspace.shared.open([file], withApplicationAt: terminal, configuration: NSWorkspace.OpenConfiguration())
    }

    func copyCommand() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(terminalCommand, forType: .string)
    }
}

struct MenuView: View {
    @ObservedObject var engine: Engine
    var inWindow = false
    @Environment(\.openWindow) private var openWindow
    @State private var showSettings = false

    var dot: Color {
        switch engine.phase {
        case .ready: return .green
        case .loading, .settingUp: return .orange
        case .failed: return .red
        case .stopped: return .secondary
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle().fill(dot).frame(width: 9, height: 9)
                Text(engine.phase.text).font(.headline)
                Spacer()
                if engine.phase == .ready { Text(engine.url).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
            }

            if !engine.hasRuntime && engine.phase != .settingUp {
                Text("第一次使用要先安装 Python 运行环境（约 1 GB）。").font(.callout)
                Button("安装运行环境") { engine.setUp() }
            } else {
                Picker("模型", selection: $engine.modelPath) {
                    ForEach(engine.modelChoices) { Text($0.label).tag($0.path) }
                }
                .disabled(engine.phase.busy)
                if !engine.modelPath.isEmpty && !FileManager.default.fileExists(atPath: engine.modelPath + "/mtp/model.safetensors") {
                    Text("这个模型没有 MTP 头，只能普通解码（约 20 tok/s）。选标着 MTP 的模型约快 30%。")
                        .font(.caption).foregroundStyle(.orange)
                }
                HStack {
                    Button("其他文件夹…") { engine.chooseFolder() }.disabled(engine.phase.busy)
                    Button("刷新列表") { engine.refreshModels() }.disabled(engine.phase.busy)
                    Spacer()
                    if engine.phase.busy {
                        Button(engine.phase == .settingUp ? "取消" : "停止") { engine.stop() }
                    } else {
                        Button("启动") { engine.start() }.keyboardShortcut(.defaultAction)
                    }
                }
            }

            if !engine.lastRequest.isEmpty {
                Text(engine.lastRequest).font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary).lineLimit(3).textSelection(.enabled)
            }

            if engine.stats.requests > 0 {
                let st = engine.stats
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text("统计").font(.subheadline.bold())
                        Spacer()
                        Button("清零") { engine.stats = EngineStats() }.font(.caption)
                    }
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                        GridRow { Text("请求"); Text("\(st.requests) 次") }
                        GridRow { Text("输入"); Text("\(st.prompt) tokens（缓存命中 \(Int(st.cacheRate * 100))%）") }
                        GridRow { Text("输出"); Text("\(st.output) tokens") }
                        GridRow { Text("读入速度"); Text(String(format: "%.0f tok/s", st.prefillSpeed)) }
                        GridRow { Text("生成速度"); Text(String(format: "%.1f tok/s", st.decodeSpeed)) }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
            }

            Divider()
            Button("在终端打开 Claude Code…") { engine.openClaude() }
                .disabled(engine.phase != .ready && engine.phase != .loading)
            Button("复制终端命令") { engine.copyCommand() }
            Button("查看日志") {
                openWindow(id: "log")
                NSApp.activate(ignoringOtherApps: true)
            }

            DisclosureGroup("设置", isExpanded: $showSettings) {
                Form {
                    TextField("端口", value: $engine.port, format: .number.grouping(.never))
                    Toggle("MTP 加速（模型带 MTP 头时）", isOn: $engine.mtp)
                    Picker("思考", selection: $engine.thinking) {
                        Text("跟随请求").tag("auto")
                        Text("总是").tag("on")
                        Text("关闭").tag("off")
                    }
                    Picker("最大上下文", selection: $engine.maxContext) {
                        Text("32K").tag(32768)
                        Text("64K").tag(65536)
                        Text("96K").tag(98304)
                        Text("128K").tag(131072)
                        Text("256K（模型原生上限）").tag(262144)
                    }
                    if engine.phase == .ready && engine.contextLimit < engine.maxContext {
                        Text("内存只够 \(engine.contextLimit / 1024)K，已自动降低。").font(.caption).foregroundStyle(.orange)
                    }
                    Text("改动在下次启动时生效。上下文越长，读入越慢：256K 读满约 1 小时。").font(.caption).foregroundStyle(.secondary)
                }
                .disabled(engine.phase.busy)
                .padding(.top, 4)
            }

            Divider()
            if !inWindow {
                Button("打开主窗口") {
                    openWindow(id: "main")
                    NSApp.activate(ignoringOtherApps: true)
                }
            }
            Button("退出 big2flash") { NSApp.terminate(nil) }
            if inWindow { Spacer() }
        }
        .buttonStyle(.borderless)
        .padding(14)
        .frame(width: inWindow ? 320 : 380)
    }
}

struct LogView: View {
    @ObservedObject var engine: Engine

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(engine.log.enumerated()), id: \.offset) { i, line in
                        Text(line).font(.system(.caption, design: .monospaced)).textSelection(.enabled).id(i)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
            }
            .onChange(of: engine.log.count) { _, n in if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) } }
        }
        .frame(minWidth: 640, minHeight: 360)
        .toolbar {
            Button("复制全部") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(engine.log.joined(separator: "\n"), forType: .string)
            }
        }
    }
}


// Totals over every request the engine served (chat and Claude Code alike), from lines like
// "[12:00:00] prompt 17170 (cached 16960, prefill 210 in 3.6s) -> 324 tokens in 34.8s (9.3 tok/s) ...".
struct EngineStats {
    var requests = 0, prompt = 0, cached = 0, prefilled = 0, output = 0
    var prefillSeconds = 0.0, decodeSeconds = 0.0
    static let pattern = try! NSRegularExpression(
        pattern: #"prompt (\d+) \(cached (\d+), prefill (\d+) in ([\d.]+)s\) -> (\d+) tokens in ([\d.]+)s"#)

    mutating func add(logLine line: String) {
        let ns = line as NSString
        guard let m = Self.pattern.firstMatch(in: line, range: NSRange(location: 0, length: ns.length)) else { return }
        let f = (1...6).map { ns.substring(with: m.range(at: $0)) }
        requests += 1
        prompt += Int(f[0]) ?? 0
        cached += Int(f[1]) ?? 0
        prefilled += Int(f[2]) ?? 0
        prefillSeconds += Double(f[3]) ?? 0
        output += Int(f[4]) ?? 0
        decodeSeconds += Double(f[5]) ?? 0
    }

    var prefillSpeed: Double { prefillSeconds > 0 ? Double(prefilled) / prefillSeconds : 0 }
    var decodeSpeed: Double { decodeSeconds > 0 ? Double(output) / decodeSeconds : 0 }
    var cacheRate: Double { prompt > 0 ? Double(cached) / Double(prompt) : 0 }
}

struct ReplyStats {
    var input = 0, cached = 0, output = 0, deltas = 0
    var started = Date(), firstToken: Date?, finished: Date?
    var prefillSeconds: Double { (firstToken ?? finished ?? Date()).timeIntervalSince(started) }
    var decodeSeconds: Double { firstToken.map { (finished ?? Date()).timeIntervalSince($0) } ?? 0 }
    // while streaming the exact count is unknown: content deltas are close to one per token
    var tokens: Int { output > 0 ? output : deltas }
    var speed: Double { decodeSeconds > 0.05 ? Double(tokens) / decodeSeconds : 0 }
    var summary: String {
        var parts = ["输入 \(input + cached)" + (cached > 0 ? "（缓存 \(cached)）" : ""),
                     "读入 " + String(format: "%.1fs", prefillSeconds)]
        if input > 0 && firstToken != nil && prefillSeconds > 0.05 {
            parts[1] += String(format: "（%.0f tok/s）", Double(input) / prefillSeconds)
        }
        parts.append((finished == nil ? "约 " : "") + "输出 \(tokens)")
        if speed > 0 { parts.append(String(format: "%.1f tok/s", speed)) }
        return parts.joined(separator: " · ")
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String
    var text: String
    var thinking = ""
    var stats: ReplyStats?
}

// Talks to the engine's /v1/messages endpoint with streaming, like Claude Code does.
@MainActor final class Chat: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input = ""
    @Published var think = false
    @Published var running = false
    @Published var error = ""
    private var task: Task<Void, Never>?
    private var pendingText = "", pendingThinking = "", pendingDeltas = 0
    private var lastFlush = Date.distantPast

    private func flush(_ i: Int) {
        if !pendingText.isEmpty { messages[i].text += pendingText; pendingText = "" }
        if !pendingThinking.isEmpty { messages[i].thinking += pendingThinking; pendingThinking = "" }
        if pendingDeltas > 0 { messages[i].stats?.deltas += pendingDeltas; pendingDeltas = 0 }
        lastFlush = Date()
    }

    func send(port: Int) {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !running else { return }
        input = ""
        error = ""
        messages.append(ChatMessage(role: "user", text: text))
        let history = messages.filter { !$0.text.isEmpty }.map { ["role": $0.role, "content": $0.text] }
        messages.append(ChatMessage(role: "assistant", text: "", stats: ReplyStats()))
        let i = messages.count - 1
        var body: [String: Any] = ["model": "big2flash", "max_tokens": 8192, "stream": true, "messages": history]
        if think { body["thinking"] = ["type": "enabled", "budget_tokens": 4000] }
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/messages")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "content-type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        req.timeoutInterval = 1800
        running = true
        task = Task {
            do {
                let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                if let http = resp as? HTTPURLResponse, http.statusCode != 200 {
                    error = "引擎返回 \(http.statusCode)"
                }
                for try await line in bytes.lines {
                    guard line.hasPrefix("data: "), let data = line.dropFirst(6).data(using: .utf8),
                          let ev = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
                    if ev["type"] as? String == "error" {
                        error = (ev["error"] as? [String: Any])?["message"] as? String ?? "引擎出错"
                        break
                    }
                    if let u = (ev["message"] as? [String: Any])?["usage"] as? [String: Any] {
                        messages[i].stats?.input = u["input_tokens"] as? Int ?? 0
                        messages[i].stats?.cached = u["cache_read_input_tokens"] as? Int ?? 0
                    }
                    if ev["type"] as? String == "message_delta",
                       let n = (ev["usage"] as? [String: Any])?["output_tokens"] as? Int {
                        messages[i].stats?.output = n
                    }
                    if let delta = ev["delta"] as? [String: Any] {
                        let t = delta["text"] as? String ?? delta["thinking"] as? String
                        if t != nil {
                            if messages[i].stats?.firstToken == nil { messages[i].stats?.firstToken = Date() }
                            pendingDeltas += 1
                        }
                        // batch UI updates (≤20/s): a redraw per token re-lays out the whole reply
                        if let t = delta["text"] as? String { pendingText += t }
                        if let t = delta["thinking"] as? String { pendingThinking += t }
                        if Date().timeIntervalSince(lastFlush) > 0.05 { flush(i) }
                    }
                }
            } catch {
                if !Task.isCancelled { self.error = "连不上引擎：\(error.localizedDescription)" }
            }
            flush(i)
            messages[i].stats?.finished = Date()
            if messages[i].text.isEmpty && messages[i].thinking.isEmpty { messages.remove(at: i) }
            running = false
        }
    }

    var totals: (input: Int, output: Int) {
        messages.compactMap(\.stats).reduce((0, 0)) { ($0.0 + $1.input + $1.cached, $0.1 + $1.tokens) }
    }

    func stop() { task?.cancel() }

    func clear() {
        stop()
        messages.removeAll()
        error = ""
    }
}

struct Bubble: View {
    let message: ChatMessage

    var rendered: AttributedString {
        (try? AttributedString(markdown: message.text,
                               options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(message.text)
    }

    var body: some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 60) }
            VStack(alignment: .leading, spacing: 6) {
                if !message.thinking.isEmpty {
                    DisclosureGroup("思考过程") {
                        Text(message.thinking).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    .font(.caption)
                }
                if message.text.isEmpty && message.thinking.isEmpty {
                    ProgressView().controlSize(.small)
                } else if !message.text.isEmpty {
                    Text(rendered).textSelection(.enabled)
                }
            }
            .padding(10)
            .overlay(alignment: .bottomLeading) {
                if let st = message.stats, st.firstToken != nil || st.finished != nil {
                    TimelineView(.periodic(from: .now, by: 0.5)) { _ in
                        Text(st.summary).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    .offset(x: 10, y: 18)
                }
            }
            .padding(.bottom, message.stats == nil ? 0 : 16)
            .background(message.role == "user" ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.10),
                        in: RoundedRectangle(cornerRadius: 10))
            if message.role != "user" { Spacer(minLength: 60) }
        }
    }
}

struct ChatView: View {
    @ObservedObject var chat: Chat
    @ObservedObject var engine: Engine

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if chat.messages.isEmpty {
                            Text(engine.phase == .ready ? "引擎已就绪，在下面输入消息开始对话。"
                                 : "先在左边选模型并点「启动」，变成「运行中」后就可以在这里对话。")
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, minHeight: 200)
                        }
                        ForEach(chat.messages) { Bubble(message: $0).id($0.id) }
                    }
                    .padding(16)
                }
                .onChange(of: chat.messages.last?.text) { _, _ in
                    if let id = chat.messages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                }
            }
            if !chat.error.isEmpty {
                Text(chat.error).font(.callout).foregroundStyle(.red).padding(.horizontal, 12).padding(.bottom, 6)
            }
            if !chat.messages.isEmpty {
                let t = chat.totals
                Text("本对话：输入 \(t.input) · 输出 \(t.output) tokens").font(.caption2).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 12).padding(.bottom, 4)
            }
            Divider()
            HStack(alignment: .bottom, spacing: 10) {
                TextField("输入消息，回车发送，⌥回车换行", text: $chat.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .onSubmit { chat.send(port: engine.port) }
                    .disabled(engine.phase != .ready)
                Toggle("思考", isOn: $chat.think).toggleStyle(.checkbox)
                if chat.running {
                    Button("停止") { chat.stop() }
                } else {
                    Button("发送") { chat.send(port: engine.port) }
                        .disabled(engine.phase != .ready || chat.input.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(12)
        }
        .toolbar {
            Button { chat.clear() } label: { Label("新对话", systemImage: "square.and.pencil") }
                .help("新对话")
        }
    }
}

struct MainView: View {
    @ObservedObject var engine: Engine
    @StateObject private var chat = Chat()

    var body: some View {
        HStack(spacing: 0) {
            ScrollView { MenuView(engine: engine, inWindow: true) }
                .frame(width: 320)
            Divider()
            ChatView(chat: chat, engine: engine)
        }
        .frame(minWidth: 860, minHeight: 560)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { Engine.shared.stopForQuit() }
    }
}

@main
struct Big2FlashApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @ObservedObject var engine = Engine.shared

    var icon: String {
        switch engine.phase {
        case .ready: return "bolt.circle.fill"
        case .loading, .settingUp: return "hourglass.circle"
        case .failed: return "exclamationmark.circle"
        case .stopped: return "bolt.circle"
        }
    }

    var body: some Scene {
        Window("big2flash", id: "main") {
            MainView(engine: engine)
        }
        .defaultSize(width: 1100, height: 720)
        MenuBarExtra("big2flash", systemImage: icon) {
            MenuView(engine: engine)
        }
        .menuBarExtraStyle(.window)
        Window("big2flash 日志", id: "log") {
            LogView(engine: engine)
        }
    }
}
