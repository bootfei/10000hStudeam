# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这个仓库是什么

一个人的 AI 转型学习仓库（10 年 Java → 大模型后训练 / RL 系统），包含两类东西，规则完全不同：

1. **学习练习**（根目录 `main*.py`）：按 `阶段0-PyTorch手感手册-v2.md` 手打的 PyTorch 练习。**不要替用户写这些代码**：只解释概念、解释报错、引导定位 bug，先让用户自己定位。完整背景和协作偏好见 `项目背景-AI训练工程师转型.md` 第九节。
2. **`steam-tracker/`**：Steam 风格的学习时长 / 打卡工具 “ai-studeam”。这是工具不是学习项目，可以直接改代码。

回复用中文。用户周日是主日（敬拜），任何计划都不排周日。学习节奏：周一到周五每天 1 小时，周六 5 小时（每周约 10 小时）。长期目标：2 年 1000 小时，7 年 10000 小时。

## 命令

学习练习：系统 Python 3.10 + torch 2.2，没有 venv / requirements。

```bash
python3 main03.py
```

steam-tracker 没有构建、没有测试框架。改完后检查脚本语法：

```bash
python3 -c "import re;s=open('steam-tracker/index.html').read();open('/tmp/page.js','w').write(re.search(r'<script>(.*)</script>',s,re.S).group(1))" && node --check /tmp/page.js
```

## steam-tracker 架构

- **发布形态**：`index.html` 是 claude.ai Artifact 的源文件，线上地址 https://claude.ai/artifact/DNdrzbZ41NYggPAc6wzkau 。用 Artifact 工具以同一文件路径重新发布即可更新（会保留 capabilities：`db`、`sample`、`downloads`）。文件按 Artifact 外壳写：没有 `<!doctype>/<html>/<body>`，开头直接是 `<title>` 和 `<style>`。
- **本地直接打开不能用**：数据和 Claude 调用都走 `window.claude.use(...)`，本地没有。要在本地跑，需要在页面前注入一个模拟 `window.claude`（内存版 db + 返回固定 JSON 的 sample）再用 `python3 -m http.server` 起服务测试。
- **数据在云端 db，不在页面里**。集合：
  - `games/{id}`：一款“游戏”=一个科目，含 `chapters[]`（每章 `questions[]`，题目带 `by: 'me'|'claude'` 和 `right/wrong` 计数）和 `dlcs[]`（一次练习 = `{date, minutes}`）。
  - `days/{YYYY-MM-DD}`：每日关卡，`targetMin`、`tasks[]`（可关联 `gameId/chapterId`）、`attempts[]`（每日测验记录）。
  - `weeks/{周一日期}`：周计划的复盘说明。
  - `meta/settings`：起始日期、长期目标、`dailyTargetMin` + 按星期覆盖的 `weekdayTargetMin`（周六 300）、`restDays: [0]`、`quizCount/quizPass`、正在计时的 `running`。
  - Claude 在会话里改数据用 ArtifactData 工具，写已有文档要带 `if_version`。`seed.js` 只是初始题库的来源，页面不会加载它（初始数据是用 ArtifactData 写进去的）。
- **状态同步的坑**：`onSnapshot` 会整体替换本地的 `games` / `days` 对象。所以 `saveGame(g)` / `saveDay(d)` 以传入的对象为准并放回本地镜像；测验状态只存 id，每次从最新数据里取。每个文档的写入经 `put()` 排队，一次只写一个。不要在快照回调里重绘正在输入的弹窗。
- **通关规则**（`dayStatus`）：时长（当天所有 DLC 分钟数）≥ 目标，且任务至少一条并全部勾完，且当天有一次测验通过。主日不设目标、不打断连续天数（`streak`）。
- **Claude 调用（`sample`）都在页面里**：`claudeGrade`（批改，`partial` 不算对）、章节出题、每日测验出题（只考当天已勾完的任务）、建议任务、“复盘这周，排下周”。都要求返回 JSON，并用 `C<编号>` 引用章节。
- 查看器里 `alert/confirm/prompt` 不可用，弹窗统一用页面内的 `ask()`。
