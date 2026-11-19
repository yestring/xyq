# 第一阶段数据处理编码提示词

请先读取 START_HERE.md、docs/ALGORITHM_SPEC.md、docs/DATA_DICTIONARY.md、docs/ACCEPTANCE_CHECKLIST.md 和 config/defaults.json。以ALGORITHM_SPEC.md为唯一算法正文；如配置或辅助说明冲突，先报告，不自行改业务范围。文件路径均相对本项目目录。不要假装已读取无法访问的文件。

系统名称：银行客户现金流预测与智能资产配置辅助决策系统 V1.0。最新要求覆盖此前讨论稿：正常消费预测；年龄主导K=3；财富起步(0.80,0.20)、财富增长(0.65,0.35)、财富安享(0.85,0.15)；仅两比例，无预算与三资产拆分。

现在授权你只实现第一阶段数据层，不做聚类、检索、神经网络、网页或数据库服务。

先检查本地Python环境及库可用性，提出最小依赖并记录实际版本；先列计划再开始实现。在本项目创建最小可测试Python结构，源文件放src/，检查放tests/，生成结果放artifacts/。不要删除或覆盖docs、config、data/raw、data/reference、examples。

实现读取校验、字符串外围单引号清理、整数分金额、raw/normal两视图、原始记录确定first_seen、正常交易确定target、180天面板及observed掩码。生成规格第04节的四个数据产物。

先用examples/toy_walkthrough.json的raw_events测试小样本，按其中coverage覆盖0至5日；这只是fixture，不改变真实数据覆盖0至179。随后处理完整原始CSV。数据守恒以data/reference/normal_data_audit.json为准：原始594643行、正常587443行、正常总金额1870843256分、4112客户、有效日720521、有效零日151903、首现前19639格、起点差异188客户。原始hash必须不变。

完成验收清单中的数据层项目，展示面板少量首尾记录及上述统计，不打印全数据。用非零进程退出码表达阻断错误。不要预生成所谓预测结果。

交付要求：只实现本阶段；遵守原始数据不变和信息截止日。使用相对路径。给出实际修改的文件、运行命令、真实执行日志、验收结果及失败原因。未执行的检查明确写未执行，不能编造通过结果。记录到 DEVELOPMENT_PROGRESS.md，区分smoke与full。不要自动推进下一阶段，等待我的明确指令。
