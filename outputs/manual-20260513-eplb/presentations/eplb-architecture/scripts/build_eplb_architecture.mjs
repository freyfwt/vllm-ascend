import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(
  "/Users/freyfwt/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json",
);
const sharp = require("sharp");

const OUT_DIR =
  "/Users/freyfwt/Projects/vllm-ascend/outputs/manual-20260513-eplb/presentations/eplb-architecture/output";
const W = 1920;
const H = 1080;

const C = {
  ink: "#17202A",
  muted: "#526071",
  faint: "#EDF1F5",
  bg: "#FAFBFC",
  blue: "#1C6FB7",
  blueSoft: "#EAF4FF",
  green: "#1E8E6F",
  greenSoft: "#EAF7F1",
  amber: "#B36B13",
  amberSoft: "#FFF3DE",
  purple: "#7354C4",
  purpleSoft: "#F1EDFF",
  red: "#C94A3A",
  redSoft: "#FCEDEA",
  line: "#CAD3DF",
  white: "#FFFFFF",
  graySoft: "#F5F7FA",
};

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function rect(x, y, w, h, fill, stroke = "none", sw = 1, rx = 8, extra = "") {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function line(x1, y1, x2, y2, color = C.line, width = 2, extra = "") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" marker-end="url(#arrow)" ${extra}/>`;
}

function pathArrow(d, color = C.line, width = 2, extra = "") {
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" marker-end="url(#arrow)" ${extra}/>`;
}

function textLines(x, y, lines, opts = {}) {
  const {
    size = 20,
    lh = 1.28,
    color = C.ink,
    weight = 500,
    anchor = "start",
    family = "Inter, PingFang SC, Hiragino Sans GB, Microsoft YaHei, Arial, sans-serif",
  } = opts;
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}">` +
    lines
      .map((l, i) => {
        const dx = i === 0 ? 0 : 0;
        const dy = i === 0 ? 0 : size * lh;
        if (typeof l === "string") {
          return `<tspan x="${x}" dx="${dx}" dy="${dy}">${esc(l)}</tspan>`;
        }
        const attrs = [
          `x="${x}"`,
          `dx="${dx}"`,
          `dy="${dy}"`,
          l.size ? `font-size="${l.size}"` : "",
          l.weight ? `font-weight="${l.weight}"` : "",
          l.color ? `fill="${l.color}"` : "",
        ]
          .filter(Boolean)
          .join(" ");
        return `<tspan ${attrs}>${esc(l.text)}</tspan>`;
      })
      .join("") +
    `</text>`;
}

function label(x, y, w, h, text, color, fill = C.white) {
  return [
    rect(x, y, w, h, fill, color, 1.5, 999),
    textLines(x + w / 2, y + h / 2 + 6, [text], {
      size: 17,
      color,
      weight: 700,
      anchor: "middle",
    }),
  ].join("");
}

function panel(x, y, w, h, title, accent, fill, subtitle = "") {
  return [
    rect(x, y, w, h, fill, accent, 2, 8, 'filter="url(#shadow)"'),
    rect(x, y, w, 42, accent, accent, 0, 8),
    rect(x, y + 34, w, 12, accent, accent, 0, 0),
    textLines(x + 18, y + 28, [{ text: title, size: 23, weight: 800, color: C.white }], {
      size: 23,
      color: C.white,
      weight: 800,
    }),
    subtitle
      ? textLines(x + w - 18, y + 27, [subtitle], {
          size: 15,
          color: C.white,
          weight: 700,
          anchor: "end",
        })
      : "",
  ].join("");
}

function card(x, y, w, h, title, lines, accent, fill = C.white, opts = {}) {
  const titleSize = opts.titleSize || 20;
  const bodySize = opts.bodySize || 17;
  const bodyY = y + 48;
  return [
    rect(x, y, w, h, fill, accent, 1.7, 8),
    rect(x, y, 8, h, accent, accent, 0, 8),
    textLines(x + 22, y + 28, [{ text: title, size: titleSize, weight: 800, color: accent }], {
      size: titleSize,
      color: accent,
      weight: 800,
    }),
    textLines(x + 22, bodyY, lines, {
      size: bodySize,
      color: opts.bodyColor || C.ink,
      weight: 500,
      lh: opts.lh || 1.28,
    }),
  ].join("");
}

const parts = [];
parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`);
parts.push(`<defs>
  <filter id="shadow" x="-6%" y="-8%" width="112%" height="118%">
    <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#1A2A3A" flood-opacity="0.12"/>
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#526071"/>
  </marker>
  <marker id="arrowRed" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="${C.red}"/>
  </marker>
</defs>`);
parts.push(rect(0, 0, W, H, C.bg, "none", 0, 0));

parts.push(textLines(60, 54, [{ text: "vLLM Ascend EPLB 实现原理：构造接口复用 + Ascend 动态控制闭环", size: 38, weight: 850 }], {
  size: 38,
  weight: 850,
}));
parts.push(
  textLines(62, 88, [
    "重点：不是直接复用主仓 EplbState.step() 完整控制器，而是在主仓 MoE/EPLB 构造语义上接入 Ascend 自研采集、规划和 D2D 换权重链路。",
  ], { size: 20, color: C.muted, weight: 500 }),
);
parts.push(label(1445, 30, 415, 40, "运行时控制权在 Ascend EPLB 链路", C.red, C.redSoft));

// Upstream semantics band.
parts.push(panel(60, 115, 1800, 170, "A. 主仓 vLLM：提供 EPLB 构造接口语义", C.blue, C.blueSoft));
parts.push(label(1470, 126, 350, 30, "不承担 Ascend 运行时重排控制", C.red, C.white));
parts.push(card(88, 168, 510, 88, "1. 开关与配置语义", [
  "`ParallelConfig.enable_eplb` / `EPLBConfig`",
  "`num_redundant_experts` 进入 MoE 构造参数",
], C.blue, C.white, { bodySize: 16 }));
parts.push(card(680, 168, 525, 88, "2. 模型构造路径", [
  "Qwen3 MoE、DeepSeek MoE 等读取 enable_eplb",
  "继续向 `FusedMoE(...)` 传 EPLB 相关参数",
], C.blue, C.white, { bodySize: 16 }));
parts.push(card(1285, 168, 535, 88, "3. FusedMoE 基础能力", [
  "expert_map / logical expert / physical expert",
  "redundant expert 与 supports_eplb 的基础抽象",
], C.blue, C.white, { bodySize: 16 }));
parts.push(line(598, 212, 680, 212, C.blue, 2.2));
parts.push(line(1205, 212, 1285, 212, C.blue, 2.2));

// Main panels.
parts.push(panel(60, 320, 500, 610, "B. Ascend 配置与进程初始化", C.green, C.greenSoft, "control plane"));
parts.push(panel(605, 320, 600, 610, "C. 模型构造与 AscendFusedMoE", C.amber, C.amberSoft, "model/runtime"));
parts.push(panel(1250, 320, 610, 610, "D. 动态 EPLB 控制闭环", C.purple, C.purpleSoft, "online loop"));

// Left cards.
parts.push(card(86, 375, 448, 82, "1. 用户入口：additional_config", [
  '`eplb_config`: dynamic_eplb / expert_map_path',
  "heat_interval / algorithm_interval / redundant / policy",
], C.green, C.white, { bodySize: 15.5, lh: 1.22 }));
parts.push(card(86, 472, 448, 88, "2. AscendConfig -> EplbConfig", [
  "默认值与合法性校验；record_path 会置 dynamic_eplb",
  "dynamic_eplb 需 `DYNAMIC_EPLB`",
  "或 `EXPERT_MAP_RECORD` 环境变量",
], C.green, C.white, { bodySize: 15.5, lh: 1.2 }));
parts.push(card(86, 575, 448, 88, "3. 平台 patch：允许子进程再生子进程", [
  "`patch/platform/__init__.py` 条件导入",
  "`patch_multiproc_executor`: WorkerProc daemon=False",
], C.green, C.white, { bodySize: 15.5, lh: 1.2 }));
parts.push(card(86, 678, 448, 96, "4. NPUModelRunner.__init__", [
  "dynamic=True 时创建 `D2DExpertWeightLoader`",
  "`EplbProcess/EplbWorker` + `EplbUpdator`",
  "`Manager().dict`: moe_load / expert_maps + Queue",
], C.green, C.white, { bodySize: 15.5, lh: 1.18 }));
parts.push(card(86, 790, 448, 86, "5. 独立通信组 dynamic_eplb", [
  "`init_ascend_model_parallel` 创建 EP-like rank group",
  "用于 moe_load all-gather 与权重 D2D P2P",
], C.green, C.white, { bodySize: 15.5, lh: 1.22 }));
parts.push(label(112, 890, 396, 26, "当前动态 EPLB 路径在 v1；v2 model_runner 显式不支持", C.green, C.white));

// Middle cards.
parts.push(card(632, 375, 548, 78, "1. load_model 接入主仓构造语义", [
  "若 dynamic_eplb 或 static expert_map_path：",
  "`parallel_config.enable_eplb = True`，再调用 `get_model()`",
], C.amber, C.white, { bodySize: 16, lh: 1.2 }));
parts.push(card(632, 468, 548, 78, "2. 主仓 MoE 构造函数保持不改", [
  "模型代码看到 enable_eplb=True",
  "`FusedMoE(...)` 构造点携带 redundant/eplb 参数",
], C.amber, C.white, { bodySize: 16, lh: 1.2 }));
parts.push(card(632, 561, 548, 88, "3. OOT CustomOp 替换实际实现", [
  '`register_ascend_customop`: "FusedMoE" -> `AscendFusedMoE`',
  "所以主仓模型文件不需要为 Ascend 动态 EPLB 改写",
], C.amber, C.white, { bodySize: 16, lh: 1.22 }));
parts.push(card(632, 665, 548, 112, "4. init_eplb_config 初始化 EPLB 数据面", [
  "`global_expert_map`: 全局 logical expert -> rank/local slot",
  "`_expert_map`: 本 rank logical -> local slot",
  "`log2phy`: 推理期 logical -> physical expert",
  "`global_redundant_expert_num` / `local_num_experts`",
], C.amber, C.white, { bodySize: 15.5, lh: 1.16 }));
parts.push(card(632, 793, 548, 106, "5. forward_impl 统计真实负载", [
  "router 选 logical expert；`log2phy` 路由到当前 placement",
  "MoE comm prepare / MLP apply / finalize 后返回 `expert_tokens`",
  "累计 `moe_load`: [local]；FlashLB 为 [iter, local]",
], C.amber, C.white, { bodySize: 15.5, lh: 1.16 }));

// Right cards.
parts.push(card(1276, 375, 558, 78, "1. model_register + eplb_warmup", [
  "动态挂载 get_all_moe_loads / clear_all_moe_loads / get_log2phy_map",
  "`VllmEplbAdaptor` 收集 expert 参数、CPU map、recv buffers",
], C.purple, C.white, { bodySize: 15.2, lh: 1.18 }));
parts.push(card(1276, 468, 558, 94, "2. forward_end：采集并唤醒 worker", [
  "到 `expert_heat_collection_interval - 1`：",
  "`get_rank_expert_workload()` -> dynamic_eplb all_gather",
  "写 `shared_dict['moe_load']`；`planner_q.put(1)`",
], C.purple, C.white, { bodySize: 15.2, lh: 1.15 }));
parts.push(card(1276, 579, 558, 112, "3. EplbWorker.do_update：规划 placement", [
  "读 `moe_load` + `expert_maps`；`PolicyFactory` 选择策略",
  "0 Random / 1 Default / 2 SwiftBalance / 3 FlashLB multi-stage",
  "校验 placement；生成 send/recv；打包 map + log2phy + layer_id",
], C.purple, C.white, { bodySize: 15.2, lh: 1.15 }));
parts.push(card(1276, 708, 558, 92, "4. forward_before：启动 D2D 换权重", [
  "到 `heat_interval + algorithm_interval - 1` 拉取 `block_update_q`",
  "逐层解析本 rank send/recv，设置新 log2phy，构造 P2POp",
  "`batch_isend_irecv` 异步传输专家权重 tensor",
], C.purple, C.white, { bodySize: 15.2, lh: 1.12 }));
parts.push(card(1276, 817, 558, 86, "5. D2DExpertWeightLoader：提交更新", [
  "状态 WAITING -> READY -> TRANSFERRING",
  "等待 reqs；更新 expert_map / log2phy / expert weight",
  "支持 unquantized `w13/w2` 与 W8A8 weight/scale 列表",
], C.purple, C.white, { bodySize: 15.2, lh: 1.12 }));

// Main arrows.
parts.push(line(310, 457, 310, 472, C.green, 2));
parts.push(line(310, 560, 310, 575, C.green, 2));
parts.push(line(310, 663, 310, 678, C.green, 2));
parts.push(line(310, 774, 310, 790, C.green, 2));
parts.push(pathArrow("M 534 726 C 585 726, 585 414, 632 414", C.green, 2.5));

parts.push(line(906, 453, 906, 468, C.amber, 2));
parts.push(line(906, 546, 906, 561, C.amber, 2));
parts.push(line(906, 649, 906, 665, C.amber, 2));
parts.push(line(906, 777, 906, 793, C.amber, 2));
parts.push(pathArrow("M 1548 256 L 1548 300 L 1225 300 L 1225 605 L 1180 605", C.blue, 2.3, 'stroke-dasharray="8 6"'));
parts.push(pathArrow("M 1180 848 C 1226 848, 1226 512, 1276 512", C.purple, 2.2));
parts.push(label(1195, 835, 120, 28, "moe_load", C.purple, C.white));
parts.push(pathArrow("M 1555 453 L 1555 468", C.purple, 2));
parts.push(pathArrow("M 1555 562 L 1555 579", C.purple, 2));
parts.push(pathArrow("M 1555 691 L 1555 708", C.purple, 2));
parts.push(pathArrow("M 1555 800 L 1555 817", C.purple, 2));
parts.push(pathArrow("M 1276 860 C 1215 860, 1215 721, 1180 721", C.red, 2.4));

// Small details callouts.
parts.push(label(650, 905, 510, 26, "static map: 构造期载入；dynamic: 周期性在线重排", C.amber, C.white));
parts.push(label(1294, 905, 522, 26, "计时: heat_interval -> algo_interval -> per-layer D2D", C.purple, C.white));

// Bottom conclusion band.
parts.push(rect(60, 955, 1800, 92, C.white, C.line, 1.5, 8, 'filter="url(#shadow)"'));
parts.push(rect(60, 955, 10, 92, C.red, C.red, 0, 8));
parts.push(textLines(88, 982, [{ text: "一页结论", size: 22, weight: 850, color: C.red }], {
  size: 22,
  color: C.red,
  weight: 850,
}));
parts.push(textLines(210, 980, [
  "vLLM Ascend EPLB = 主仓 EPLB 构造接口 + Ascend 自研动态重排执行链路；主仓只让 MoE 进入 EPLB 语义，运行时采集/规划/D2D 由 Ascend 完成。",
  "数据闭环：logical expert -> physical expert -> local token load -> global workload -> placement plan -> D2D weight swap -> new expert_map/log2phy。",
  "代码索引：ascend_config.py, model_runner_v1.py, parallel_state.py, ops/fused_moe.py, eplb_utils.py, eplb_updator.py, eplb_worker.py, eplb_device_transfer_loader.py, vllm_adaptor.py",
], { size: 17, color: C.ink, weight: 520, lh: 1.28 }));

parts.push(`</svg>`);

const svg = parts.join("\n");
const svgPath = path.join(OUT_DIR, "vllm_ascend_eplb_architecture.svg");
const pngPath = path.join(OUT_DIR, "vllm_ascend_eplb_architecture.png");
const png2xPath = path.join(OUT_DIR, "vllm_ascend_eplb_architecture_2x.png");

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(svgPath, svg, "utf8");
await sharp(Buffer.from(svg)).png().toFile(pngPath);
await sharp(Buffer.from(svg.replace(`width="${W}" height="${H}"`, `width="${W * 2}" height="${H * 2}"`)))
  .png()
  .toFile(png2xPath);

console.log(JSON.stringify({ svgPath, pngPath, png2xPath }, null, 2));
