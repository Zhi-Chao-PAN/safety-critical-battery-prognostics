# 贡献指南（Contributing Guide）

👋 感谢你对这个项目的兴趣！我们欢迎任何形式的贡献。

---

## 📋 目录

- [行为准则](#行为准则)
- [我可以贡献什么？](#我可以贡献什么)
- [开发环境搭建](#开发环境搭建)
- [代码贡献流程](#代码贡献流程)
- [代码规范](#代码规范)
- [提交信息规范](#提交信息规范)
- [Pull Request指南](#pull-request指南)

---

## 行为准则

### 我们承诺
为了营造开放和友好的环境，我们承诺：
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 不可接受的行为
- 使用性化的语言或图像
- 恶意评论或人身攻击
- 公开或私下骚扰
- 未经许可发布他人的私人信息
- 其他不专业或不恰当的行为

---

## 我可以贡献什么？

### 🐛 报告Bug
发现问题了？请提交Issue，包含：
- 清晰的标题和描述
- 复现步骤
- 预期行为和实际行为
- 环境信息（Python版本、依赖版本等）
- 错误日志和截图

### ✨ 提出新功能
有好想法？请提交Feature Request，说明：
- 功能的用途和场景
- 期望的实现方式
- 可能的替代方案

### 📝 改进文档
文档永远可以更好！你可以：
- 修复拼写错误
- 补充说明和示例
- 翻译文档
- 改进代码注释

### 💻 贡献代码
我们欢迎：
- 修复Bug
- 实现新功能
- 优化性能
- 改进代码结构
- 添加测试用例

---

## 开发环境搭建

### 1. Fork并克隆项目
```bash
# 在GitHub上Fork项目后
git clone https://github.com/YOUR_USERNAME/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics

# 添加上游仓库
git remote add upstream https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
```

### 2. 创建虚拟环境
```bash
# 使用venv
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. 安装依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -e ".[dev]"
```

### 4. 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_chronos_model.py -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

---

## 代码贡献流程

### 1. 同步最新代码
```bash
git checkout main
git pull upstream main
```

### 2. 创建功能分支
```bash
# 使用清晰的分支名
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/issue-description
```

### 3. 进行开发
- 编写代码
- 添加或更新测试
- 更新文档（如需要）

### 4. 提交更改
```bash
# 查看变更
git status
git diff

# 暂存文件
git add .

# 提交（遵循提交信息规范）
git commit -m "feat: add new feature"
```

### 5. 推送分支
```bash
git push origin feature/your-feature-name
```

### 6. 创建Pull Request
- 在GitHub上创建Pull Request
- 使用PR模板
- 关联相关Issue
- 请求代码审查

---

## 代码规范

### Python代码风格
- 遵循[PEP 8](https://peps.python.org/pep-0008/)
- 使用类型注解（Type Hints）
- 最大行长度：100字符
- 使用Google风格的Docstring

### 示例代码
```python
from typing import List, Optional
import torch

def predict_rul(
    capacity_sequence: torch.Tensor,
    context_length: int = 100,
    device: Optional[str] = None
) -> torch.Tensor:
    """预测电池剩余使用寿命（RUL）.

    Args:
        capacity_sequence: 容量序列张量，形状为 (batch_size, seq_len)
        context_length: 使用的上下文长度
        device: 计算设备，默认为自动选择

    Returns:
        预测的RUL张量，形状为 (batch_size,)

    Examples:
        >>> import torch
        >>> seq = torch.randn(1, 200)
        >>> rul = predict_rul(seq)
        >>> print(rul)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 实现逻辑...
    pass
```

### 测试规范
- 为新功能编写单元测试
- 测试文件放在`tests/`目录
- 使用`pytest`框架
- 测试命名：`test_<function_name>`

---

## 提交信息规范

我们使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type类型
- `feat`: 新功能
- `fix`: 修复Bug
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 重构（既不是新增功能，也不是修复bug）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具链相关

### 示例
```
feat(models): add new transformer model

- Add transformer architecture for RUL prediction
- Add positional encoding
- Update tests for new model

Closes #123
```

---

## Pull Request指南

### PR创建前检查
- [ ] 代码已通过所有测试
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 代码遵循项目规范
- [ ] 提交信息符合规范
- [ ] 与main分支已同步

### PR模板
请使用以下模板创建PR：

```markdown
## 描述
清晰描述这个PR做了什么。

## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新
- [ ] 代码重构

## 测试
- [ ] 已添加单元测试
- [ ] 所有测试通过
- [ ] 已在本地验证

## 相关Issue
Closes #<issue-number>

## 截图/演示
（如适用）
```

### 代码审查
- 我们会尽快审查PR
- 可能会要求一些修改
- 保持沟通，耐心回应

---

## 获得帮助

如果在贡献过程中遇到问题：
1. 查看文档和FAQ
2. 搜索现有的Issues和Discussions
3. 开启新的Discussion提问
4. 在相关Issue下留言

---

## 认可贡献者

我们会在项目中记录所有贡献者。感谢你的参与！🎉

---

## 许可证

通过贡献代码，你同意你的贡献将在项目的MIT许可证下发布。
