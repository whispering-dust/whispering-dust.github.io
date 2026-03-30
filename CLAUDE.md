# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pnpm dev        # 启动开发服务器（localhost:4859）
pnpm build      # SSG 构建到 dist/
pnpm serve      # 预览构建产物
```

> 注意：GitHub Actions 工作流使用 `npm`，本地开发用 `pnpm`（项目有 pnpm-lock.yaml）。

推送到 `main` 分支后，GitHub Actions 自动构建并部署到 `https://whispering-dust.github.io/`。

## 架构

本项目是基于 **Valaxy 0.28.4** + **valaxy-theme-yun** 的静态博客。

### 核心配置

| 文件 | 用途 |
|---|---|
| `site.config.ts` | 站点元信息：标题、作者、favicon、社交链接、赞助等 |
| `valaxy.config.ts` | 框架与主题配置：主题、导航、背景、公告、页脚等 |

两个配置通过 `defu` 合并，用户配置优先级最高。

### 内容

- 文章放在 `pages/posts/*.md`，frontmatter 控制标题、日期、分类、标签等
- 分类和标签无需预注册，写在 frontmatter 里自动聚合
- `<!-- more -->` 标记前的内容作为列表页摘要

### 静态资源

- `public/` 下的文件原样复制到 `dist/`，通过根路径 `/` 引用
- 背景图：`public/images/bg.jpg`（亮色）和 `public/images/bg_dark.jpg`（暗色）
- 头像：`public/images/avatar.jpg`

### 自定义扩展

- `styles/index.scss` — 自定义 CSS
- `styles/css-vars.scss` — 覆盖 CSS 变量（主题色等）
- `components/` — 自定义/覆盖 Vue 组件（按文件名覆盖主题组件）
- `layouts/` — 自定义布局

### yun 主题关键配置项（`valaxy.config.ts` → `themeConfig`）

```ts
bg_image: { enable, url, dark, opacity }   // 背景图
notice: { enable, content }                 // 公告（支持 HTML）
pages: [{ name, url, icon, color }]         // 侧边栏导航图标
nav: [{ text, link, icon }]                 // 顶部导航栏
banner: { enable, title }                   // 首页横幅标语
footer: { since, beian }                    // 页脚配置
```
