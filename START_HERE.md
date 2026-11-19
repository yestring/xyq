# 给新手的使用说明

这是一份可独立交给其他编码 AI 的算法交接包。系统名称为“银行客户现金流预测与智能资产配置辅助决策系统 V1.0”。它包含完整原始交易表和实现规格，不包含已开发的系统、已训练模型或实验成绩。

## 你现在怎么使用

1. 如果收到ZIP，先解压，保持整个 BankSim_AI_Coding_Handoff 目录结构。
2. 用能读取本地项目文件的编码AI打开这个目录。若AI只能聊天，先上传 docs/ALGORITHM_SPEC.md、docs/DATA_DICTIONARY.md、docs/ACCEPTANCE_CHECKLIST.md 和 config/defaults.json；数据文件上传不了时，保留在本地，让AI生成读取你本地路径的程序，不要把几十万行复制进对话。
3. 第一次发送 prompts/00_READ_AND_CONFIRM.md 的全文。它只要求读取与确认方案，不写代码。无需再上传此前的讨论稿或旧V1.1。
4. 检查AI是否正确复述：剔除欺诈金额、年龄权重最大、三类客群与三对比例、两个检索路径、按时间隔离。若它要增加预算、三资产组合或收入预测，请要求回到本规格。
5. 确认后，再发送 prompts/01_DATA_STAGE.md。这才是第一轮编码授权，只实现数据层。阶段完成后检查真实输出是否满足验收，不要只看AI说“完成”。
6. 后续按02、03、04、05顺序，每次只授权一个阶段。遇到报错，把完整报错、执行命令和当前阶段名称交给AI，使用 prompts/ERROR_RECOVERY.md。不要连续要求AI盲目重写整个项目。

## 文件阅读顺序

| 文件 | 给谁读 | 作用 |
| --- | --- | --- |
| docs/ALGORITHM_SPEC.md | 编码AI及你 | 唯一算法正文，18节，优先级最高 |
| docs/ALGORITHM_SPEC.docx | 你 | 同内容Word阅读副本 |
| docs/DATA_DICTIONARY.md | 编码AI | CSV字段、可见性和样本说明 |
| config/defaults.json | 编码AI | 首版参数副本，不能凌驾正文 |
| docs/ACCEPTANCE_CHECKLIST.md | 你和编码AI | 分阶段可核验条件 |
| prompts/00至05 | 你复制给AI | 逐阶段任务边界 |
| data/raw/bs140513_032310.csv | 程序读取 | 作者完整原始交易表 |
| data/reference/normal_data_audit.json | 数据验收时读取 | 已核验总量，只作核验，不当模型特征 |
| examples/toy_walkthrough.json | 编码AI | 虚构小例子，包含预期计算结果 |
| manifest.json | 校验工具 | 文件大小和SHA-256清单 |

## 交接纪律

后续实现代码可放在解压后的项目根目录中，但不要覆盖本包规范或raw数据。采用Python及NumPy、pandas、scikit-learn、PyTorch、pyarrow、pytest；由编码AI先检测环境、核对兼容版本并锁定实际依赖，不直接套用过时版本号。当前没有requirements锁文件，也没有偷偷创建可运行项目。

原始CSV的类别值带外围单引号，不能把客户ID转为数字。网络边表不需要，不在包内。旧的全180天客户汇总不作模型输入，避免未来信息泄漏。新的数据层须从raw生成面板。

首版默认H=7、L=28、W=28、C=14，候选步长7。年龄/均消费/波动/活跃度权重0.60/0.20/0.10/0.10。财富起步80/20、财富增长65/35、财富安享85/15。稳健比例不是现金储备。

先完成算法阶段再讨论软件界面与软著系统方案。本交接包不代表已经完成软件，也不承诺登记或毕业审核结果。
