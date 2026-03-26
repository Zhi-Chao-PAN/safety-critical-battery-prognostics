# GitHub 仓库设置指南

这份指南帮助你把项目上传到GitHub并获得更多star。

---

## 第一步：创建GitHub仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `safety-critical-battery-prognostics`
   - **Description**: 微-宏时间尺度解耦的物理信息神经网络在电池无分布寿命预测中的应用
   - **Public/Private**: 选择 **Public**（开源项目才能获得star）
   - **不要**勾选"Initialize this repository with..."（因为本地已有代码）
3. 点击 **Create repository**

---

## 第二步：上传代码

在本地项目目录执行：

```bash
# 1. 初始化git（如果还没有）
git init

# 2. 添加所有文件
git add .

# 3. 提交
git commit -m "Initial commit: Micro-Macro Time-Scale Decoupling Battery Prognostics"

# 4. 关联远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/safety-critical-battery-prognostics.git

# 5. 推送到GitHub
git branch -M main
git push -u origin main
```

---

## 第三步：配置仓库设置

### 3.1 基本信息设置

在仓库页面点击 **Settings**，然后：

1. **General** → **Features**
   - ✅ 勾选 **Discussions**（开启讨论区）
   - ✅ 勾选 **Projects**（项目看板）
   - ✅ 勾选 **Wiki**（可选）

2. **General** → **Pull Requests**
   - ✅ 勾选 **Allow squash merging**
   - ✅ 勾选 **Automatically delete head branches**

### 3.2 填写仓库描述

在仓库首页（Code标签页），点击右侧的 **About** 旁边的 ⚙️ 图标：

- **Description**:
  ```
  微-宏时间尺度解耦的电池寿命预测系统 | Micro-Macro Time-Scale Decoupling for Battery RUL Prediction
  ```

- **Website**（可选，没有就留空）:

- **Topics**（很重要！添加这些标签）:
  ```
  battery
  prognostics
  physics-informed
  pytorch
  deep-learning
  time-series
  uncertainty-quantification
  iso26262
  bms
  electric-vehicles
  ```

---

## 第四步：开启Issue模板和Discussions

### 4.1 Issue模板（可选但推荐）

在仓库 **Settings** → **General** → **Features** 中已开启Issues。

可以在 `.github/` 目录创建Issue模板（可选）。

### 4.2 Discussions分类

在 **Discussions** 标签页，点击 **New discussion** 右侧的 ⚙️ → **Categories**，添加：

- 📢 Announcements（公告）
- 💡 Ideas（想法）
- ❓ Q&A（问答）
- 🛠️ Show and tell（展示）
- 📚 Documentation（文档）

---

## 第五步：社交媒体分享（可选但推荐）

### 5.1 README社交预览

GitHub会自动显示README的内容，我们已经优化好了。

### 5.2 分享到相关社区

可以分享到：
- Twitter/X (#battery #AI #MachineLearning)
- Reddit (r/MachineLearning, r/EV)
- 知乎（电池、机器学习话题）
- 相关的微信群/QQ群
- 学术社区（ResearchGate, arXiv）

---

## 第六步：维护与互动

### 获得更多Star的技巧：

1. **及时回复Issues** - 有人提问就快速回应
2. **欢迎贡献** - 保持CONTRIBUTING.md更新
3. **定期更新** - 保持项目活跃
4. **使用Discussions** - 建立社区氛围
5. **感谢贡献者** - 在README中致谢

### 发布Release（可选）

当有重大更新时：
1. 点击 **Releases** → **Draft a new release**
2. 填写版本号（如 `v1.0.0`）
3. 写更新日志
4. 点击 **Publish release**

---

## 仓库检查清单

上传后检查：

- [ ] 代码已成功推送
- [ ] README显示正常
- [ ] About描述和Topics已添加
- [ ] Discussions已开启
- [ ] 仓库是Public的
- [ ] 分支是main（不是master）

---

## 常见问题

### Q: 上传时提示大文件超过100MB？
A:
- 使用Git LFS处理大文件，或
- 把大的checkpoint和数据文件加到`.gitignore`

### Q: 如何让项目更受欢迎？
A:
- 写清晰的README
- 提供快速开始教程
- 及时响应社区
- 持续改进和更新

### Q: 可以修改用户名和仓库名吗？
A: 可以，在GitHub Settings里修改，但要注意更新所有链接。

---

## 需要帮助？

如有问题，参考GitHub官方文档：https://docs.github.com/
