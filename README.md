# big2flash

在 Mac 上本地跑 Qwen3.8-27B 的 MLX 推理引擎和 macOS App，用模型自带的 MTP 头做推测解码。

- `engine/server.py`：Anthropic / OpenAI 兼容的本地服务（MTP 推测解码、流水线、线性注意力状态回滚），说明见 `engine/SERVER.md`
- `engine/hwprobe.py`：检测本机芯片、内存和带宽，推荐最快的模型和配置
- `app/`：macOS App（Swift），`app/build.sh` 打包
- `tests/bench/`：各引擎同条件测速脚本和原始数据
- `report/`：测速报告和长图，`bench/RESULTS.md` 是历次结果

测试机：MacBook Pro M2 Max 96GB。模型权重不在仓库里。
