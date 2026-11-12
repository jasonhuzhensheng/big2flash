// Renders the app icon (1024 px PNG) to the path given as the first argument.
import AppKit

let size = 1024.0
let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()
let rect = NSRect(x: 100, y: 100, width: 824, height: 824)
let shape = NSBezierPath(roundedRect: rect, xRadius: 185, yRadius: 185)
NSGradient(starting: NSColor(red: 0.10, green: 0.12, blue: 0.22, alpha: 1),
           ending: NSColor(red: 0.20, green: 0.30, blue: 0.62, alpha: 1))!.draw(in: shape, angle: 90)
let config = NSImage.SymbolConfiguration(pointSize: 520, weight: .bold)
    .applying(.init(paletteColors: [NSColor(red: 1.0, green: 0.78, blue: 0.25, alpha: 1)]))
let bolt = NSImage(systemSymbolName: "bolt.fill", accessibilityDescription: nil)!.withSymbolConfiguration(config)!
let b = bolt.size
bolt.draw(in: NSRect(x: (size - b.width) / 2, y: (size - b.height) / 2, width: b.width, height: b.height))
img.unlockFocus()
let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
