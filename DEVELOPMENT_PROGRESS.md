# 第一阶段：数据处理与日消费面板构建

日期：2026-11-15（Asia/Shanghai）。规格版本：1.0.0。状态：本阶段完成，等待用户明确授权下一阶段。

依据为 `docs/ALGORITHM_SPEC.md`，参考第一阶段提示词、数据字典、D01—D15验收清单、默认配置、参考审计及toy示例。本阶段仅实现数据读取、校验、面板、可见性约束及验收；没有聚类、检索、预测网络、配置界面、数据库服务或训练结果。

## 1. 实施计划与文件

执行顺序：检测环境 → 创建项目虚拟环境 → 实现最小数据模块 → toy运行及异常边界检查 → 完整CSV处理 → 全量产物回读与D01—D15验收 → 原始资料完整性核对。

本次新建文件如下。没有修改任何交接包原有文件。

| 文件或目录 | 用途 |
| --- | --- |
| `src/banksim/__init__.py` | 规格版本及包标识 |
| `src/banksim/data.py` | 严格CSV解析、字段校验、分类字符串清理、Decimal整数分、原始行号、raw/normal防御性副本视图、重复及属性冲突证据 |
| `src/banksim/panel.py` | raw首次可见日、属性来源登记、完整客户日网格、normal金额与交易笔数、null/0掩码 |
| `src/banksim/visibility.py` | 普通/人工开放窗口、属性来源限制、连续历史及标签完整性检查；不生成实验ID划分 |
| `src/banksim/quality.py` | 从真实输入计算统计，随后与参考审计比较，验证独立守恒条件 |
| `src/banksim/data_stage.py` | toy/full命令入口、Parquet生成与回读、质量报告、阻断错误退出码2 |
| `tests/test_data.py` | 数据校验、金额、重复、冲突、CLI错误退出码及全量产物检查 |
| `tests/test_visibility.py` | 隐藏/未来属性隔离、欺诈记录属性来源、缺口、冷窗准入及标签边界 |
| `tests/conftest.py` | 从pytest真实执行结果生成D01—D15状态；未执行的全量检查不记为通过 |
| `pytest.ini` | src导入路径、测试目录及验收标记 |
| `requirements-data.lock.txt` | 实测环境依赖版本，包括必要的间接依赖 |
| `.gitignore` | 排除虚拟环境、缓存和生成产物；文件仍保存在本地 |
| `DEVELOPMENT_PROGRESS.md` | 本交付记录 |
| `.venv/` | 项目独立Python虚拟环境，未修改全局环境 |
| `artifacts/data/` | 本阶段四个正式数据产物 |
| `artifacts/smoke/data/` | 明确标为smoke的toy数据产物 |
| `artifacts/acceptance_data.json`、`artifacts/acceptance_smoke.json` | 全量和smoke的真实验收结果、测试节点及耗时 |
| `artifacts/logs/` | 执行日志、JUnit报告、环境记录、资料完整性和初次失败证据 |
| `artifacts/test-tmp/`、`artifacts/test-tmp-smoke/` | pytest的隔离夹具及预期失败报告，均不属于正式数据产物 |

原始文件保留不变。客户和商户ID、邮编为字符串；amount在Parquet中为decimal128(20,2)，amount_cents为int64，日目标为可空int64。三个Parquet的schema metadata均保存spec_version及原始data_hash。

重复记录报告原始行号且保留，不自动去重。客户属性冲突保留全部相关来源行，登记表不选取冲突属性，窗口读取返回ATTRIBUTE_CONFLICT；命令行遇冲突退出2，不发布成功数据集。窗口属性仅检查开放区间，不让隐藏或未来记录恢复属性。当前真实CSV未发现重复或属性冲突。

## 2. 环境与依赖

原项目没有虚拟环境。PATH中的python指向WindowsApps占位入口；找到已有可用解释器：

`C:/Users/qqq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`

该基础运行时为Python 3.12.14，原有NumPy 2.3.5、pandas 3.0.1，没有pyarrow和pytest。基础运行时未被修改。

实际执行环境为 `.venv/Scripts/python.exe`，Python **3.12.14，Windows 11，64位AMD64**。`include-system-site-packages=false`，pip为25.0.1。

| 直接依赖 | 实际版本 |
| --- | --- |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| pyarrow | 26.0.1 |
| pytest | 9.2.0 |

必要的间接依赖：colorama 0.4.6、iniconfig 2.3.1、packaging 26.3、pluggy 1.7.0、Pygments 2.21.0、python-dateutil 4.0.0、tzdata 2026.6。完整版本见 `requirements-data.lock.txt` 和 `artifacts/logs/environment.json`。

`pip check`实际输出：`No broken requirements found.` 没有安装scikit-learn、PyTorch或GPU依赖。环境检查时可用磁盘空间为282,357,374,976字节；成功写入并回读了全部Parquet。

## 3. 实际执行命令及问题处理

以下命令从项目根目录执行。环境初始化使用已存在的基础解释器；后续命令均使用项目虚拟环境。

```powershell
& 'C:/Users/qqq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir numpy pandas pyarrow pytest
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 1 --timeout 20 numpy pandas pyarrow pytest
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -LiteralPath 'requirements-data.lock.txt' -Encoding ASCII
.\.venv\Scripts\python.exe -m pip check
```

环境初始化经历的实际问题：

1. venv的ensurepip在沙箱中写临时目录被拒绝，退出1。显式运行ensurepip也得到PermissionError；经工具权限提升后完成，输出 `Successfully installed pip-25.0.1`。
2. 沙箱中pip安装失败；权限提升后的官方PyPI下载仍出现 `SSL: UNEXPECTED_EOF_WHILE_READING`，未安装成功。随后使用清华HTTPS PyPI镜像，在项目.venv中成功安装上表依赖，退出0；没有禁用TLS验证。
3. pytest在沙箱中访问自己创建的临时目录时遇到WinError 5；经工具权限提升后，测试在项目artifacts目录内正常执行。没有修改系统全局Python或为数据阶段安装训练依赖。

实际数据与测试命令：

```powershell
.\.venv\Scripts\python.exe src/banksim/data_stage.py --mode toy 2>&1 | Tee-Object -FilePath 'artifacts/logs/toy-final.log'
.\.venv\Scripts\python.exe -m pytest -m 'not full' --basetemp=artifacts/test-tmp --acceptance-report=artifacts/acceptance_smoke.json --junitxml=artifacts/logs/pytest-smoke.xml
.\.venv\Scripts\python.exe src/banksim/data_stage.py --mode full 2>&1 | Tee-Object -FilePath 'artifacts/logs/full.log'
.\.venv\Scripts\python.exe -m pytest --basetemp=artifacts/test-tmp --acceptance-report=artifacts/acceptance_data.json --junitxml=artifacts/logs/pytest-full.xml 2>&1 | Tee-Object -FilePath 'artifacts/logs/pytest-full.log'
.\.venv\Scripts\python.exe -m pytest -m 'not full' --basetemp=artifacts/test-tmp-smoke --acceptance-report=artifacts/acceptance_smoke.json --junitxml=artifacts/logs/pytest-smoke.xml 2>&1 | Tee-Object -FilePath 'artifacts/logs/pytest-smoke-final.log'
```

带管道的执行包装在命令之后保存 `$LASTEXITCODE` 并以该值退出，未将Tee-Object成功误当成程序成功。

| 执行 | 真实结果 | 证据 |
| --- | --- | --- |
| 初次toy | 退出2；4项示例对照相符，但null掩码断言因pandas普通bool与可空boolean类型不同而误报 | `artifacts/logs/toy.log`、`toy-initial-report.json` |
| 修复后toy | PASS，退出0，程序耗时1.002秒；15条raw、14条normal、正常31,000分、3客户×6日 | `artifacts/logs/toy-final.log`、`artifacts/smoke/data/quality_report.json` |
| 初次smoke测试 | 临时目录权限导致部分检查未完成，退出1 | `artifacts/logs/pytest-smoke-initial.log` |
| 权限修复后、全量之前的smoke | 46 passed、6 deselected，2.66秒，退出0 | 当次终端真实输出；后续最终smoke补充两项检查并覆盖最终smoke日志 |
| full数据处理 | PASS，退出0，程序耗时14.742秒；47项参考对照及11项独立条件全部相符 | `artifacts/logs/full.log`、`artifacts/data/quality_report.json` |
| 最终全部测试 | **54 passed in 4.52s**，退出0 | `artifacts/logs/pytest-full.log`、`pytest-full.xml`、`artifacts/acceptance_data.json` |
| 更新后的最终smoke报告 | 48 passed、6 deselected，4.19秒，退出0；未运行全量检查的验收项明确记NOT_EXECUTED | `artifacts/logs/pytest-smoke-final.log`、`pytest-smoke.xml`、`artifacts/acceptance_smoke.json` |

布尔掩码误报已通过明确保存普通bool及按值比较解决，没有改动消费值或预期统计。另补充了属性冲突命令行退出码、乱序原始记录属性追溯检查。最终验收依据为全量54项测试，不能用smoke报告替代全量验收。

其他检查：计算manifest列出的全部18个文件的大小和SHA-256，全部一致，见 `artifacts/logs/source-integrity.json`。原目录不是Git仓库，`git status --short` 返回 `fatal: not a git repository`；没有初始化仓库，此情况不影响数据验收。

## 4. D01—D15逐项状态

状态由pytest运行生成，完整测试节点、执行阶段与耗时见 `artifacts/acceptance_data.json`。最终没有失败或未执行的数据层验收项。

| ID | 最终状态 | 实际证据 |
| --- | --- | --- |
| D01 | 通过 | 原始CSV处理前、处理后、验收时SHA-256相同，并与manifest及audit一致 |
| D02 | 通过 | 实际解析594,643行、10个原字段、4,112客户、step 0—179；保存连续raw_row_id 1—594643 |
| D03 | 通过 | 外侧空白/一对单引号夹具；内部引号保留，ID及邮编仍为字符串 |
| D04 | 通过 | 0、0.10、1.23、1234.56、0.29分别精确转换为0、10、123、123456、29分；Decimal列回读 |
| D05 | 通过 | 非法精度、负值、NaN/Infinity、非法枚举、越界、空ID、缺字段及畸形CSV全部阻断；CLI实际退出2且不生成部分Parquet |
| D06 | 通过 | 重复夹具保留3条记录，报告原始行1/2，正常日金额仍为4,000分，没有去重 |
| D07 | 通过 | 年龄、类型及两者冲突夹具均记录来源行；冲突客户属性为空，读取报ATTRIBUTE_CONFLICT；CLI实际退出2 |
| D08 | 通过 | 第0日欺诈50、第3日正常10：raw起点0，前四日目标分为[0,0,0,1000]且均observed |
| D09 | 通过 | 全量raw/normal/fraud数量、金额、零额交易及面板正常金额守恒；欺诈记录保留 |
| D10 | 通过 | 全量740,160个唯一客户日，720,521有效日、151,903有效零日、19,639个不可见null；Parquet目标类型int64且可空 |
| D11 | 通过 | 从持久化事件重新聚合raw/normal首次日，188名客户延迟，最大120日；登记起点取raw |
| D12 | 通过 | 首现前null，最后交易之后至179仍为可见0；仅有欺诈的客户也保留 |
| D13 | 通过 | 开放窗口无raw记录时年龄/类型为U；隐藏和未来记录不能恢复属性 |
| D14 | 通过 | 开放窗口仅欺诈记录仍可提供年龄/类型及来源日，消费为0；零额正常记录计笔数 |
| D15 | 通过 | 缺行、重复日、observed=0、目标null、标签越179和冷窗不足均拒绝；标签恰到179合法；推理历史检查允许无未来标签 |

## 5. 全量实际值／预期值对照

以下实际值全部来自程序解析和聚合，再与参考审计比较。实现没有把预期值写入actual统计。质量报告还包含客户年龄/类型分布、商户和类别等，共47项参考比较，以及11项独立约束。

| 统计 | 实际值 | 预期值 | 一致 |
| --- | ---: | ---: | --- |
| 原始交易行数 | 594,643 | 594,643 | 是 |
| 正常交易行数 | 587,443 | 587,443 | 是 |
| 欺诈交易行数 | 7,200 | 7,200 | 是 |
| raw金额总计（分） | 2,253,110,373 | 2,253,110,373 | 是 |
| 正常金额总计（分） | 1,870,843,256 | 1,870,843,256 | 是 |
| 欺诈金额总计（分） | 382,267,117 | 382,267,117 | 是 |
| 面板正常金额总计（忽略null，分） | 1,870,843,256 | 1,870,843,256 | 是 |
| 客户数 | 4,112 | 4,112 | 是 |
| 面板单元数 | 740,160 | 740,160 | 是 |
| 有效客户日 | 720,521 | 720,521 | 是 |
| 有效零消费日 | 151,903 | 151,903 | 是 |
| 有效正消费日 | 568,618 | 568,618 | 是 |
| 首现前不可见null单元 | 19,639 | 19,639 | 是 |
| 正常首次日晚于raw首次日客户 | 188 | 188 | 是 |
| 首次日最大差值（天） | 120 | 120 | 是 |
| 正常零额交易 | 52 | 52 | 是 |
| 有正常交易但日金额为0的客户日 | 49 | 49 | 是 |
| 完全重复行 | 0 | 0 | 是 |
| 客户年龄冲突数 | 0 | 0 | 是 |
| 客户类型冲突数 | 0 | 0 | 是 |

原始CSV处理前和处理后的SHA-256均为：

`e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377`

## 6. 四个正式产物与示例

| 路径 | 行数／内容 | 字节数 |
| --- | --- | ---: |
| `artifacts/data/events_clean.parquet` | 594,643行，原10字段+raw_row_id+amount_cents | 6,915,090 |
| `artifacts/data/customer_registry.parquet` | 4,112行，首次可见日、属性、来源日/原始行号及属性状态 | 80,076 |
| `artifacts/data/daily_panel.parquet` | 740,160行，customer/step/observed/target_cents/raw_tx_count/normal_tx_count | 1,699,908 |
| `artifacts/data/quality_report.json` | 实际统计、参考比较、独立约束、源哈希、环境、三个Parquet校验及少量示例 | 17,198 |

三个Parquet均在写入后重新读取并逐表比较，`read_back_equal=true`；随后全量pytest再次读取持久化文件验证。产物SHA-256记录在quality_report中。

清洗记录示例（摘录字段）：raw_row_id=1，customer=C1093826151，step=0，age=4，gender=M，amount=4.55，amount_cents=455，fraud=0。

登记表示例：customer=C1000148617，first_seen_step=30，age=5，gender=M，attribute_source_step=30，attribute_source_raw_row_id=80564，attribute_status=CONSISTENT。

日面板真实示例：

| customer | step | observed | target_cents | raw_tx_count | normal_tx_count |
| --- | ---: | --- | ---: | ---: | ---: |
| C1000148617 | 0 | false | null | 0 | 0 |
| C1000148617 | 1 | false | null | 0 | 0 |
| C1000148617 | 31 | true | 0 | 0 | 0 |
| C999723254 | 177 | true | 6255 | 1 | 1 |
| C999723254 | 178 | true | 2596 | 1 | 1 |
| C999723254 | 179 | true | 3296 | 1 | 1 |

## 7. 未解决事项与阶段停止点

本阶段尚未解决的问题：**没有**。D01—D15全部通过，数据输入、规范、配置、审计与示例文件均保持原有哈希。

后续复现先使用本项目.venv；需要重建时按requirements-data.lock.txt安装本阶段锁定依赖。该机器沙箱曾限制pytest临时目录访问，因此本次成功的测试运行使用了已获准的权限提升；这是执行环境记录，不是未执行检查。

没有进行第二阶段ID切分、画像、聚类、检索或模型训练。当前只声明数据层符合本阶段验收，不声明预测效果。现停止，等待用户下一步明确指令。

---

# 第二阶段：实验划分、画像、加权K-means和固定比例

日期：2026-11-15（Asia/Shanghai）。依据用户本轮明确授权及 `prompts/02_CLUSTER_STAGE.md`。规格版本1.0.0。本阶段已完成；下文是追加记录，上述第一阶段历史保留。

## 1. 范围、实施与文件

实施顺序：读取规范和第一阶段证据 → 建立只读文件校验基线 → 检查.venv并按第一阶段锁文件约束安装scikit-learn → 实现和运行小样本边界检查 → 复用三个正式Parquet完成O/C全量实验 → 保存及重载核验 → 第二阶段验收和第一阶段回归 → 追加记录。

本轮没有重写第一阶段数据模块，没有重新生成或覆盖第一阶段Parquet、质量报告、日志、验收JSON及依赖锁文件。没有实现检索、消费预测网络、页面、数据库服务，未安装PyTorch或GPU依赖。

| 新增/修改文件 | 用途 |
| --- | --- |
| 新增 `src/banksim/splits.py` | 字符串排序+固定种子严格ID划分、普通已见客户准入、合法时间样本、开放历史与属性区间 |
| 新增 `src/banksim/profiles.py` | 复用第一阶段窗口读取；最近28日画像、预热资格、截尾与标准化、常数列、sqrt权重、金额尺度a |
| 新增 `src/banksim/clustering.py` | 固定K-means、稳定业务命名、FULL/AGE_ONLY/未知/企业/无历史/阻断状态、真实簇描述与稳定性诊断 |
| 新增 `src/banksim/allocation.py` | 固定三对稳健/进取比例及null状态，独立于预测金额 |
| 新增 `src/banksim/cluster_io.py` | 验证第一阶段输入hash、保存版本化Parquet/JSON/模型、配置检查及模型重载 |
| 新增 `src/banksim/cluster_stage.py` | O/C独立全量流程、样本产物、原始拟合标签、实际调用示例及报告 |
| 新增 `tests/test_stage2_unit.py` | 28项边界、合成数据及泄漏变异检查 |
| 新增 `tests/test_stage2_full.py` | 18项全量产物、划分、参数、重载、诊断和完整性检查 |
| 修改 `tests/conftest.py` | 增加 `--acceptance-stage=2`，单独汇总C01—C14/E01—E04/A01—A06，禁止第二阶段报告覆盖第一阶段位置 |
| 新增 `requirements-cluster.lock.txt` | 第二阶段完整依赖快照，保留原requirements-data.lock.txt |
| 追加 `DEVELOPMENT_PROGRESS.md` | 本阶段实际过程与结果 |

第一阶段的`EventViews`及`read_visible_window`继续复用；为每个客户建立小切片以避免反复复制全量日志。客户属性始终从开放窗口内raw记录读取，不用registry属性填补隐藏信息。registry仅供ID/首次可见日等准入核对。

## 2. 实际环境、依赖和命令

继续使用 `.venv/Scripts/python.exe`，Python 3.12.14，Windows 64位。原有12个包（含pip）均未升级，见 `artifacts/stage2/logs/environment-before.json` 和 `environment-after.json`。

| 本阶段新增依赖 | 实际版本 |
| --- | --- |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.8.0 |
| narwhals | 2.28.0 |
| cloudpickle | 3.1.2 |

NumPy 2.5.4、pandas 3.1.1、pyarrow 26.0.1、pytest 9.2.0均保持原版本。新增包均为scikit-learn及其必要传递依赖。`pip check`输出 `No broken requirements found.`。K-means及轮廓系数计算使用单线程上下文，记录随机种子，不修改全局线程环境。

实际安装及快照命令：

```powershell
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 1 --timeout 20 -c requirements-data.lock.txt scikit-learn
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -LiteralPath 'requirements-cluster.lock.txt' -Encoding ASCII
.\.venv\Scripts\python.exe -m pip check
```

使用已经成功使用过的清华HTTPS镜像及项目虚拟环境，依赖安装经工具权限提升执行；没有禁用TLS或修改全局Python。

实际实施和验收命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stage2_unit.py --basetemp=artifacts/stage2/unit-tmp --acceptance-stage=2 --acceptance-report=artifacts/stage2/acceptance-unit.json --junitxml=artifacts/stage2/logs/unit.xml -o cache_dir=artifacts/stage2/pytest-cache
.\.venv\Scripts\python.exe src/banksim/cluster_stage.py
.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_visibility.py --basetemp=artifacts/stage2/regression-tmp --acceptance-report=artifacts/stage2/acceptance-data-regression.json --junitxml=artifacts/stage2/logs/data-regression.xml -o cache_dir=artifacts/stage2/pytest-cache
.\.venv\Scripts\python.exe -m pytest tests/test_stage2_unit.py tests/test_stage2_full.py --basetemp=artifacts/stage2/acceptance-tmp --acceptance-stage=2 --acceptance-report=artifacts/stage2/acceptance-cluster.json --junitxml=artifacts/stage2/logs/cluster-acceptance.xml -o cache_dir=artifacts/stage2/pytest-cache
```

上述运行通过`Tee-Object`分别保存为`unit-initial.log`、`full-run.log`、`data-regression.log`、`cluster-acceptance.log`，均在`artifacts/stage2/logs/`。包装命令保存并返回Python的`$LASTEXITCODE`。

| 实际运行 | 结果 |
| --- | --- |
| 单元/边界smoke | 28 passed in 18.76s，退出0；没有全量数据项的验收不预填PASS |
| O/C完整产物生成 | PASS，98.475秒，退出0 |
| 第二阶段完整验收 | 46 passed in 13.37s，退出0；24个验收ID均PASS |
| 第一阶段回归 | 54 passed in 8.81s，退出0；D01—D15均PASS |
| 既有依赖比较 | 原有12个包版本全部不变 |
| 受保护文件比较 | 42个源资料、第一阶段产物及证据文件的hash全部不变 |

一次单元测试工具会话在启动时中断，返回`exec cell 47 not found`，未留下测试日志或运行结果；未将其计为已执行检查。随后重新执行，得到上表28项真实通过结果。本轮没有算法测试失败、没有为通过验收调整聚类阈值或客户归属。

## 3. O/C划分、有效样本和预热拟合

严格C：全部4,112个ID字符串排序后由NumPy default_rng(42)打乱，按floor(0.7N)、floor(0.15N)、余数得到2,878/616/618；三组两两不相交。实际名单保存在`artifacts/splits/C.json`。

普通O：具有合法训练窗口的客户4,111名，验证和测试沿用这4,111个ID。客户`C1809646275`首次可见日135，没有合法训练窗口，单列排除，不混入普通已见客户成绩。该客户恰在C训练ID名单中，因此C训练分配2,878人、实际有训练样本2,877人；并未将无样本客户冒充已评价。

| 实验 | 训练/验证/测试分配ID数 | 对应实际有样本客户数 | 预热拟合人数 | 保留预热全零人数 | 金额尺度a | 预热正日金额数 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| O | 4111 / 4111 / 4111 | 4111 / 4111 / 4111 | 3979 | 177 | 26.81 | 165697 |
| C | 2878 / 616 / 618 | 2877 / 616 / 618 | 2785 | 120 | 26.88 | 116953 |

划分前3,979名资格客户、177名全零客户只作为描述性资格审计。C实际拟合队列是其训练ID交集2,785名；O/C各自拟合分位数、均值、标准差、K-means和金额尺度。a使用相应全部训练ID的0—59日有效正日金额中位数，不限于聚类资格客户，也不是逐笔金额中位数。

| 实验/阶段 | 普通历史样本 | Lc=1 | Lc=3 | Lc=7 | Lc=14 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| O训练 | 221680 | 221680 | 221629 | 221512 | 221217 | 1107718 |
| O验证 | 98664 | — | — | — | — | 98664 |
| O测试 | 98664 | — | — | — | — | 98664 |
| C训练 | 155136 | 155136 | 155100 | 155017 | 154804 | 775193 |
| C验证 | — | 14784 | 14784 | 14784 | 14784 | 59136 |
| C测试 | — | 14832 | 14832 | 14832 | 14832 | 59328 |

训练合计是保存的可选历史模式样本数，不是未来每轮训练抽样量。后续训练仍须按规格先均匀抽客户再抽合法t，并以0.5概率截短；本阶段没有实施抽样训练。

每条样本包含experiment、partition、customer、origin_t、history_mode、Lc、history_start/end、attribute_start/end、Tobs、ell、H、target_start/end。训练t=59—112/标签60—119；验证t=119—142/标签120—149；测试t=149—172/标签150—179。普通客户在每个t检查是否已经出现；冷窗要求足够连续可见天数。目标必须完整观测，数据缺口不被拼接。

## 4. 真实三簇汇总、年龄分布与中心

以下为主种子42的实际拟合标签统计。年龄均值是0—6等级的均值，不是岁数；消费为每位客户最近28日日均正常消费的分布，单位为原数据金额单位。展示四舍五入，保存和计算使用完整精度。

| 实验 | 原始簇→业务名称 | 人数 | 占比 | 年龄等级均值 | 日均消费均值 | 日均消费中位数 | 平均CV | 平均正消费日比例 | 全零画像 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O | 0→A财富起步 | 2158 | 54.23% | 2.201112 | 28.158352 | 27.572679 | 0.963314 | 0.882348 | 0 |
| O | 2→B财富增长 | 669 | 16.81% | 2.771300 | 4.591360 | 3.077500 | 3.058784 | 0.057228 | 170 |
| O | 1→C财富安享 | 1152 | 28.95% | 4.635417 | 26.612007 | 26.905536 | 1.082804 | 0.838046 | 7 |
| C | 0→A财富起步 | 1528 | 54.87% | 2.189791 | 28.136262 | 27.602857 | 0.951838 | 0.883087 | 0 |
| C | 1→B财富增长 | 455 | 16.34% | 2.852747 | 4.465225 | 2.911071 | 3.047850 | 0.054945 | 115 |
| C | 2→C财富安享 | 802 | 28.80% | 4.649626 | 26.595189 | 26.905536 | 1.088546 | 0.837727 | 5 |

O的原始簇编号映射为0→A、2→B、1→C，C则为0→A、1→B、2→C。名称由原年龄等级均值排序得到，没有固定把原始0/1/2视作A/B/C。

| 实验/业务簇 | 年龄码0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O/A | 13 | 328 | 1029 | 788 | 0 | 0 | 0 |
| O/B | 6 | 63 | 222 | 204 | 135 | 39 | 0 |
| O/C | 0 | 0 | 0 | 0 | 599 | 374 | 179 |
| C/A | 10 | 240 | 728 | 550 | 0 | 0 | 0 |
| C/B | 4 | 39 | 150 | 123 | 106 | 32 | 1 |
| C/C | 0 | 0 | 0 | 0 | 409 | 265 | 128 |

| 实验/业务簇 | 日均消费最小 | Q25 | Q75 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| O/A | 9.661071 | 24.100446 | 31.578482 | 111.486786 |
| O/B | 0 | 0 | 7.382500 | 34.409286 |
| O/C | 0 | 23.011429 | 30.730089 | 59.469286 |
| C/A | 9.810000 | 24.169018 | 31.582143 | 64.545714 |
| C/B | 0 | 0 | 7.117500 | 34.409286 |
| C/C | 0 | 23.064821 | 30.737054 | 57.073571 |

加权标准化中心如下，列序为age_rank、log_mean、cv、positive_day_ratio。数值不是原始年龄或金额，不能直接解释为30岁/40岁。

| 实验/业务簇 | 年龄维 | log_mean维 | CV维 | 活跃度维 |
| --- | ---: | ---: | ---: | ---: |
| O/A | -0.4681479101 | 0.1843546469 | -0.0973110121 | 0.1395312059 |
| O/B | -0.1347520030 | -0.8323376682 | 0.4297143672 | -0.6201214409 |
| O/C | 0.9552189931 | 0.1380178576 | -0.0672584614 | 0.0987438381 |
| C/A | -0.4728814438 | 0.1825421418 | -0.0964918184 | 0.1371051512 |
| C/B | -0.0890068006 | -0.8501701361 | 0.4332248007 | -0.6282111675 |
| C/C | 0.9514475566 | 0.1345424180 | -0.0619423763 | 0.0951862971 |

完整分位数、标准差、年龄比例、CV和活跃度分布保存在各实验`cluster_diagnostics.json`及主报告中。

**实际解释限制：两实验中，名为“财富增长”的B簇实际日均消费明显低于另外两簇，并包含大量预热全零画像。不能据此宣称其高收入、高资产或更强消费能力。** 名称按已定年龄顺序映射，完整财富阶段画像尚未验证。未删除全零客户、未人为调整标签、未改变比例或阈值。

## 5. 状态及固定比例的实际调用

以下调用在O/C两个已拟合模型上均已执行，来源标记为`real_data_asof_call`，保存于各实验`allocation_examples.json`。

| 输入客户/切点 | 历史模式 | 实际状态 | 名称 | 稳健/进取 |
| --- | --- | --- | --- | --- |
| C1002658784，t=59 | 普通 | FULL | 财富起步客户 | 0.80 / 0.20 |
| C100045114，t=59 | 普通 | FULL | 财富增长客户 | 0.65 / 0.35 |
| C1000148617，t=59 | 普通 | FULL | 财富安享客户 | 0.85 / 0.15 |
| C1012876398，t=59 | 普通，最近28日全零 | AGE_ONLY | 财富增长客户，仅按年龄暂定 | 0.65 / 0.35 |
| C1000699316，t=119 | Lc=1，开放窗口无raw记录 | GROUP_UNKNOWN | null | null / null |
| C2075935351，t=59 | 普通，企业 | NON_PERSON | null | null / null |
| C1000148617，t=29 | 首现日30之前 | NO_HISTORY | null | null / null |

AGE_ONLY不强制同簇筛选，保留暂定说明。企业输出不提供个人画像。未知/企业/无历史均不伪造第四簇。人工冲突及缺口由第一阶段函数继续阻断；少于3个不同特征向量和不足3个非空簇两种失败夹具均得到CLUSTER_FIT_FAILED，预测及客群、比例为空，未实现或调用预测网络。

同时实际调用固定规则三次，得到(0.80,0.20)、(0.65,0.35)、(0.85,0.15)。全部比例处于[0,1]、和与1的绝对差≤1e-9，rule_version=1.0.0。改变输入记录中的预测总额不改变比例。输出schema没有预算、现金储备或三资产权重。

预热供体标签独立存于`warm_labels.parquet`，直接等于主KMeans.labels_。即使同一个客户对外调用返回AGE_ONLY，也不会改写其预热拟合标签。

## 6. 诊断与重载结果

| 诊断 | O | C |
| --- | ---: | ---: |
| 加权空间全队列轮廓系数 | 0.5200064065838039 | 0.5221484501397099 |
| 主模型惯性 | 1424.3126430255554 | 1001.1393440276329 |
| 主模型迭代数 | 4 | 9 |
| A/B年龄等级均值差 | 0.5701883075593162 | 0.6629566768310227 |
| B/C年龄等级均值差 | 1.8641162182361737 | 1.796878682414842 |
| 小簇阈值 | 39.79人 | 27.85人 |
| small_cluster | false | false |
| semantic_warning | false | false |

在各自相同队列和变换上，分别重新拟合种子42、43、44。三个两两比较的调整兰德指数ARI及业务名称一致率在O/C中均为1.0。这是本数据和既定配置下的实际稳定性，不证明真实银行泛化、投资适当性或完整财富画像。未计算或声称p<0.05。

保存后重新加载验证了：变换结果、中心、原始拟合标签、predict标签、金额尺度、客群映射及实际调用结果，O/C各7项全部一致。改变期望数据hash或配置的重载请求返回CONFIG_MISMATCH。所有正式产物记录规格版本、源数据hash、实验、拟合截止59、配置及hash、随机种子和依赖版本；模型joblib与伴随JSON通过模型文件hash绑定。

## 7. 第二阶段逐项验收及回归

完整真实测试证据见`artifacts/stage2/acceptance-cluster.json`，包含测试节点、执行结果和耗时。以下24项均通过，没有失败或未执行项。

| ID | 状态 | 证据 |
| --- | --- | --- |
| C01 | 通过 | [10,0,30,20]画像的均值、总体标准差、CV、活跃度、log_mean及低金额CV分母夹具 |
| C02 | 通过 | 全零及零额交易画像均为0；全量预热全零画像保留 |
| C03 | 通过 | 划分前3979/177；O3979/177、C2785/120；每客户一条，C与训练ID取交集；原拟合标签冻结 |
| C04 | 通过 | 修改60—179日或留出ID的金额/属性，预处理、尺度、中心和标签不变；实际a从训练正日金额核对 |
| C05 | 通过 | 1%/99%线性分位数、仅指定两列截尾、常数列在测试变化后仍输出0 |
| C06 | 通过 | 标准化差向量[1,2,3,4]加权平方距离=3.9，使用sqrt(weight) |
| C07 | 通过 | 年龄0/6为等级，U为未知，不编码7，不产生精确年龄或虚构属性 |
| C08 | 通过 | FULL使用冻结四维加权最近中心，与保存模型及真实样本调用一致 |
| C09 | 通过 | 低天数/低正消费日触发AGE_ONLY，忽略行为维度，并列按业务A/B/C，禁止强制同簇标记 |
| C10 | 通过 | 未知年龄/类型、企业及冷窗属性隔离，分别正确返回GROUP_UNKNOWN/NON_PERSON |
| C11 | 通过 | 按原年龄均值、消费均值、原始编号依次排序；O/C真实映射不同 |
| C12 | 通过 | 不足3个不同向量和不足3个非空簇均阻断，不输出成功预测或比例 |
| C13 | 通过 | 合成夹具真实触发小簇和年龄间隔告警，不移动成员；全量保留实际false结果 |
| C14 | 通过 | 全量人数、比例、年龄和消费/CV/活跃分布、轮廓系数、三种子及ARI均产出 |
| E01 | 通过 | 字符串排序、default_rng(42)、floor边界2878/616/618、实际名单互斥完整 |
| E02 | 通过 | 合法切点、标签边界、开放区间、Lc及历史连续性检查；全量持久化样本核验 |
| E03 | 通过 | 普通仅合法训练已见ID，每个较早t仍检查出现；无训练窗口客户单列 |
| E04 | 通过 | O/C独立拟合和产物；C留出ID不进入预热队列、变换或金额尺度；冷窗隐藏资料隔离 |
| A01 | 通过 | 三个FULL客群实际返回三对固定比例 |
| A02 | 通过 | AGE_ONLY保留暂定状态并返回对应比例 |
| A03 | 通过 | 未知、企业、无历史和聚类失败返回null，不增加第四簇 |
| A04 | 通过 | 比例范围、和误差≤1e-9、rule_version核对 |
| A05 | 通过 | 改变预测金额不影响比例，只有稳健/进取两比例，无预算或三资产输出 |
| A06 | 通过 | 保留描述边界；增长簇低消费真实报告，未推断收入/资产/婚姻/退休或投资最优 |

第一阶段回归单独生成`artifacts/stage2/acceptance-data-regression.json`：D01—D15全部PASS，54项测试通过。原`artifacts/acceptance_data.json`、原smoke验收以及第一阶段日志未覆盖。阶段二报告明确标记stage=2，不以D项PASS代替C/E/A验收。

## 8. 产物、完整性及停止点

| 路径 | 内容 |
| --- | --- |
| `artifacts/splits/O.json`、`C.json` | 实际ID名单、时间协议、样本统计、版本和训练ID hash |
| `artifacts/stage2/full/O/`、`C/` | 两实验独立产物目录 |
| 各实验 `samples_train/validation/test.parquet` | 合法样本及开放窗口信息 |
| 各实验 `excluded_customers.parquet` | 普通未见/无训练窗口客户单列；C无样本ID另在split统计中报告 |
| 各实验 `warm_cohort.json`、`warm_profiles.parquet` | 实际拟合名单和原始画像，含全零客户 |
| 各实验 `preprocessing.json`、`amount_scale.json` | 分位数、标准化/常数列/权重及金额尺度 |
| 各实验 `kmeans.joblib`、`cluster_bundle.json`、`group_map.json` | 聚类模型、中心、配置与映射 |
| 各实验 `warm_labels.parquet` | 主种子原始标签及43/44种子标签，供后续候选库使用 |
| 各实验 `cluster_diagnostics.json` | 三种子描述、轮廓系数及稳定性 |
| 各实验 `allocation_examples.json`、`reload_checks.json` | 实际状态/比例调用及重载一致性 |
| `artifacts/stage2/full/stage2_report.json` | 本阶段真实汇总 |
| `artifacts/stage2/full/artifact_inventory.json` | 全量产物hash清单 |
| `artifacts/stage2/acceptance-cluster.json` | 第二阶段逐项验收 |
| `artifacts/stage2/acceptance-data-regression.json` | 第一阶段回归证据 |
| `artifacts/stage2/logs/` | 独立运行日志、环境、JUnit及完整性记录 |

`artifacts/stage2/baseline.json`记录42个受保护文件的原hash，`logs/integrity-after.json`确认全部不变。第一阶段历史的原始字节另存`artifacts/stage2/progress-stage1.txt`，用于验证本文件只追加第二阶段。

本阶段预设small_cluster和semantic_warning均未触发；业务名称与完整财富画像的差异已如实解释，不是通过修改阈值消除的告警。尚未解决的实施或验收问题：**没有**。当前停止，不自动进入第三阶段。


---

# 第三阶段：历史候选、双路径检索与参考序列（2026-11-16）

实施顺序：读取第07、08节及前两阶段证据 → 环境/受保护文件基线 → toy → 检索与缓存边界 → 小批真实查询 → 独立验收及前两阶段回归 → 追加交付记录。
本阶段完成。未训练消费预测网络、未开发界面、未重新拟合真实聚类。回归测试中的微型合成拟合仅验证旧模块，未改写 O/C 模型。所有新增实验结果均位于 artifacts/stage3。

## 1. 文件及接口

| 文件 | 用途 |
| --- | --- |
| `src/banksim/candidates.py` | 紧凑日期索引、连续观测验证、分块精确距离、稳定排序/贪心及稳定 softmax |
| `src/banksim/retrieval_cache.py` | 带内容哈希的查询 JSON 与候选日期 NPZ 缓存；重载及损坏检查 |
| `src/banksim/retrieval.py` | 可见窗口、画像、双路径路由、同簇/全库重选、参考序列及追溯接口 |
| `src/banksim/retrieval_io.py` | 校验并读取前两阶段冻结产物；显式 toy 假设中心；无真实聚类拟合 |
| `src/banksim/retrieval_stage.py` | toy、确定性小批真实查询、独立扩库诊断、实际指标和缓存重放报告 |
| `tests/test_retrieval.py` | 29 个边界测试用例，覆盖 R01—R15；含独立精确枚举对照 |
| `tests/test_retrieval_full.py` | 5 个真实结果/产物集成测试，逐条核算候选、日期、金额及来源 |
| `tests/conftest.py` | 扩展第三阶段验收汇总；保留 D/C/E/A 能力，允许回归报告另存 stage3 |
| `DEVELOPMENT_PROGRESS.md` | 只追加本阶段记录，保留前两阶段原始字节前缀 |

`load_engines()` 校验并加载 O/C；调用 `engine.retrieve(customer, t, Lc=None, partition='train', method='same_group', library_mode='frozen')`。
`method='all'`为相同供体全集对照；`library_mode='rolling'`为独立扩展，缓存与主库分离，本轮仅在合成边界用例验证。返回R、r_log/r_mask、候选K/V及日期/距离/权重、实际数量、状态/原因及完整版本信息，无预测金额字段。数据缺口和配置异常抛出 DataError。
供体原始标签始终来自 warm_labels.parquet；目标画像调用原有冻结模型。self候选不要求属于预热队列；本人空库保留 NO_REFERENCE。

## 2. 环境、命令和实际执行

使用项目 `.venv/Scripts/python.exe`，Python 3.12.14，pip 25.0.1。未安装或升级依赖，未修改全局Python；torch_present=False。依赖快照与前两阶段分开保留：

```text
cloudpickle==3.1.2
colorama==0.4.6
iniconfig==2.3.1
joblib==1.6.0
narwhals==2.28.0
numpy==2.5.4
packaging==26.3
pandas==3.1.1
pluggy==1.7.0
pyarrow==26.0.1
Pygments==2.21.0
pytest==9.2.0
python-dateutil==4.0.0
scikit-learn==1.9.1
scipy==1.18.1
threadpoolctl==3.8.0
tzdata==2026.6
```

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```
已保存 environment.lock.txt；未安装或升级任何依赖

```powershell
.\.venv\Scripts\python.exe -m pip check
```
No broken requirements found

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m banksim.retrieval_stage --mode sample 2>&1 | Tee-Object artifacts/stage3/logs/sample-run.log
```
82次调用、首次缓存命中0，内部耗时16.846673秒

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q --basetemp artifacts/stage3/pytest-unit-authorized -o cache_dir=artifacts/stage3/pytest-cache --acceptance-stage=3 --acceptance-report artifacts/stage3/acceptance-unit.json 2>&1 | Tee-Object artifacts/stage3/logs/unit.log
```
29 passed in 6.12s

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_retrieval_full.py -q --basetemp artifacts/stage3/pytest-acceptance -o cache_dir=artifacts/stage3/pytest-cache-retrieval --acceptance-stage=3 --acceptance-report artifacts/stage3/acceptance-retrieval.json 2>&1 | Tee-Object artifacts/stage3/logs/retrieval-acceptance.log
```
34 passed in 18.09s

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_retrieval_full.py -q --basetemp artifacts/stage3/pytest-acceptance-final -o cache_dir=artifacts/stage3/pytest-cache-final --acceptance-stage=3 --acceptance-report artifacts/stage3/acceptance-final.json 2>&1 | Tee-Object artifacts/stage3/logs/retrieval-final.log
```
补查候选索引重载/损坏后：34 passed in 18.82s

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_visibility.py -q --basetemp artifacts/stage3/pytest-data-regression -o cache_dir=artifacts/stage3/pytest-cache-data --acceptance-stage=1 --acceptance-report artifacts/stage3/regression-data.json 2>&1 | Tee-Object artifacts/stage3/logs/data-regression.log
```
54 passed in 7.63s

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stage2_unit.py tests/test_stage2_full.py -q --basetemp artifacts/stage3/pytest-cluster-regression -o cache_dir=artifacts/stage3/pytest-cache-cluster --acceptance-stage=2 --acceptance-report artifacts/stage3/regression-cluster.json 2>&1 | Tee-Object artifacts/stage3/logs/cluster-regression.log
```
46 passed in 13.12s

耗时取自实际程序/pytest输出；不包括人工等待和启动开销。三阶段验收与回归曾并发执行，耗时不是串行总时长。
初次toy调用因ndarray不能直接由已有JSON序列化函数处理而失败，改为tolist后重跑得到正确结果；日志分别为logs/toy-initial.log、logs/toy.log。首次pytest在沙箱临时目录清理时报WinError 5，未认作通过；通过环境授权机制重跑，原失败日志保留在logs/unit-initial.log。另一次只读检查漏写UTF-8编码导致UnicodeDecodeError，改为显式UTF-8后完成，未写入数据。当前目录不含.git，因此git status不可用；文件保护验证使用SHA-256。

## 3. 逐项验收及回归

| 项目 | 实际状态 | 证据 |
| --- | --- | --- |
| R01 | PASS | Tobs=0/1/14/15 路由及零消费日；真实 NO_HISTORY/SELF/CROSS 追溯（5个关联测试） |
| R02 | PASS | V末日等于 cutoff 纳入、下一日排除；训练 t 与主评价119（3个关联测试） |
| R03 | PASS | 本人 V末日≤t−ell，下一日排除；人工窗口起点同步限制（3个关联测试） |
| R04 | PASS | t=61/62/89 实际候选数0/1/4；键末日27、34、41、48（1个关联测试） |
| R05 | PASS | a=0与偏移a=3的网格；ell=1至28逐个检查，K/V缺日不入库（2个关联测试） |
| R06 | PASS | 冻结原始标签与预热供体队列；C训练ID隔离、目标本人排除；同簇/全库同一全集（7个关联测试） |
| R07 | PASS | Lc=1/3/7/14同步限制Q、属性、画像；隐藏完美匹配排除；无raw属性未知；缺日阻断（9个关联测试） |
| R08 | PASS | 距离、字符串donor_id、key_end稳定排序；重复调用顺序/权重一致（1个关联测试） |
| R09 | PASS | 跨客户每供体≤2、K+V区间互不重叠；本人可取5条且历史条目可相互重叠（2个关联测试） |
| R10 | PASS | 同簇过滤后不足m重排全库，不锁定原条目；真实O/C独立m1500扩库诊断（2个关联测试） |
| R11 | PASS | toy仅2条不复制；空库全0参考与mask=0；本人空库不跨客户回退（3个关联测试） |
| R12 | PASS | toy两距离0、权重0.5、R=[20,30]；真实V原金额加权后log1p逐条复算（2个关联测试） |
| R13 | PASS | 拒绝tau=0/负数/NaN/Inf；大距离softmax有限、非负、归一化（7个关联测试） |
| R14 | PASS | 实验/客户/切点/模式/Lc/参数/版本/截止日缓存隔离；冻结/滚动分离；重载一致、损坏拒绝（4个关联测试） |
| R15 | PASS | block=1/2/17/2048与独立精确枚举的完整ID、排序、距离、R一致；真实块上限2048（5个关联测试） |

第一阶段回归：D01=PASS、D02=PASS、D03=PASS、D04=PASS、D05=PASS、D06=PASS、D07=PASS、D08=PASS、D09=PASS、D10=PASS、D11=PASS、D12=PASS、D13=PASS、D14=PASS、D15=PASS。

第二阶段回归：C01=PASS、C02=PASS、C03=PASS、C04=PASS、C05=PASS、C06=PASS、C07=PASS、C08=PASS、C09=PASS、C10=PASS、C11=PASS、C12=PASS、C13=PASS、C14=PASS、E01=PASS、E02=PASS、E03=PASS、E04=PASS、A01=PASS、A02=PASS、A03=PASS、A04=PASS、A05=PASS、A06=PASS。

最终第三阶段34测试通过；第一阶段54测试通过；第二阶段46测试通过。各JSON保留具体测试节点、阶段、耗时及失败原因字段。没有未执行的本阶段必需验收项。

## 4. toy实际结果

Q=[30,20]，状态AGE_ONLY → ALL_DONORS；D1/D2的K均为第0—1日，V均为第2—3日，V分别[10,20]、[30,40]。实际距离[0,0]，权重[0.5,0.5]，R=[20,30]。m=5但仅返回2条，不复制。H=2/L=2/stride=1只用于给定示例。

## 5. 真实查询与抽样

采用确定性覆盖抽样：分别对O/C、train/validation/test、每个可用历史模式/Lc，在第二阶段合法样本表按(customer字符串, origin_t)排序，取前两名不同客户及各自最早合法t；跨客户查询调用同簇和全库两方法，本人查询调用一次。主样本72次调用、40个不同查询键：O14个、C26个。另加O/C各4个边界诊断及各1个独立扩库敏感性调用，共82次。不是随机总体估计，也不是全量检索实验。
主参数m=5未出现扩库。为用真实数据覆盖扩库分支，另外取训练样本中按(customer,t)排序首个FULL且业务码B、Lc=14的客户，使用独立m=1500查询；默认配置文件未修改，指标不合并。O同簇实际接受1114条，C为753条，均重新从全库选择1500条。

### q010：O / C1000148617 / t=59 / Lc=7

集合=main_sample；状态=FULL / CROSS / SAME_GROUP；候选数=5；截止日=59。
Q=[39.56, 19.03, 134.16, 11.74, 38.93, 40.79, 44.38]

R=[33.63499818000038, 21.857652430759856, 28.906493967920696, 21.356586094247167, 39.10560161634283, 32.337161057476244, 26.70374922014681]

| 供体 | K日期 | V日期 | 距离 | 权重 |
| --- | --- | --- | --- | --- |
| C1934886086 | 21—27 | 28—34 | 0.104371126891 | 0.209783795096 |
| C774836418 | 15—21 | 22—28 | 0.153635626861 | 0.199699343551 |
| C2117938225 | 41—47 | 48—54 | 0.154416553862 | 0.199543453818 |
| C168361707 | 30—36 | 37—43 | 0.170356531935 | 0.19638795167 |
| C331622323 | 28—34 | 35—41 | 0.179577151762 | 0.194585455864 |

候选ID、完整K/V金额及全部条目见query_examples.json和query_results.jsonl；超过5条的诊断在此仅展示前5条。

### q023：O / C100045114 / t=62 / Lc=None

集合=boundary_diagnostic；状态=AGE_ONLY / SELF / SELF；候选数=1；截止日=62。
Q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 351.61, 0.0, 0.0, 0.0]

R=[0.0, 0.0, 0.0, 0.0, 125.87, 0.0, 0.0]

| 供体 | K日期 | V日期 | 距离 | 权重 |
| --- | --- | --- | --- | --- |
| C100045114 | 0—27 | 28—34 | 4.29995020383 | 1 |

候选ID、完整K/V金额及全部条目见query_examples.json和query_results.jsonl；超过5条的诊断在此仅展示前5条。

### q025：O / C1000148617 / t=29 / Lc=None

集合=boundary_diagnostic；状态=NO_HISTORY / NONE / NO_HISTORY；候选数=0；截止日=29。
Q=[]

R=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

### q026：O / C1005806982 / t=59 / Lc=14

集合=expansion_sensitivity_m1500；状态=FULL / CROSS / EXPANDED_ALL；候选数=1500；截止日=59。
Q=[0.0, 0.0, 0.0, 44.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84, 0.0, 0.0, 0.0, 0.0]

R=[8.811021673842065, 7.975874127856361, 6.929742565839862, 7.579527045548332, 8.578952107090375, 7.205197822703135, 8.926917031955814]

| 供体 | K日期 | V日期 | 距离 | 权重 |
| --- | --- | --- | --- | --- |
| C2128227866 | 37—50 | 51—57 | 0.0225060117568 | 0.00221356633371 |
| C1051985937 | 36—49 | 50—56 | 0.0778236897291 | 0.00209444219502 |
| C1376256048 | 30—43 | 44—50 | 0.0778522143154 | 0.00209438245278 |
| C1124718683 | 8—21 | 22—28 | 0.0780980167429 | 0.00209386771175 |
| C1152092987 | 29—42 | 43—49 | 0.0783121822884 | 0.00209341932544 |

候选ID、完整K/V金额及全部条目见query_examples.json和query_results.jsonl；超过5条的诊断在此仅展示前5条。

### q053：C / C1004300450 / t=119 / Lc=7

集合=main_sample；状态=FULL / CROSS / SAME_GROUP；候选数=5；截止日=119。
Q=[39.39, 11.23, 59.54, 36.34, 16.98, 3.58, 20.85]

R=[39.40586541394495, 25.497686165847387, 26.191435197622276, 28.134779942718218, 19.83473121747917, 22.389379194355143, 28.299610841206743]

| 供体 | K日期 | V日期 | 距离 | 权重 |
| --- | --- | --- | --- | --- |
| C788955662 | 91—97 | 98—104 | 0.0553378277504 | 0.207015687708 |
| C1720950881 | 22—28 | 29—35 | 0.0849303035202 | 0.200979336773 |
| C150273597 | 91—97 | 98—104 | 0.0978916179096 | 0.198391189532 |
| C1263461120 | 78—84 | 85—91 | 0.101630620437 | 0.197650789414 |
| C1272188168 | 105—111 | 112—118 | 0.110206555756 | 0.195962996574 |

候选ID、完整K/V金额及全部条目见query_examples.json和query_results.jsonl；超过5条的诊断在此仅展示前5条。

### q054：C / C1004300450 / t=119 / Lc=7

集合=main_sample；状态=FULL / CROSS / ALL_DONORS；候选数=5；截止日=119。
Q=[39.39, 11.23, 59.54, 36.34, 16.98, 3.58, 20.85]

R=[28.754892201370875, 28.117542786873294, 23.345129249124156, 48.574236783110564, 30.17367225391466, 53.18568136749923, 27.52204308780222]

| 供体 | K日期 | V日期 | 距离 | 权重 |
| --- | --- | --- | --- | --- |
| C788955662 | 91—97 | 98—104 | 0.0553378277504 | 0.201994602713 |
| C673994225 | 92—98 | 99—105 | 0.0608638162648 | 0.200881461294 |
| C424790668 | 68—74 | 75—81 | 0.0621770504306 | 0.200617830039 |
| C907777350 | 105—111 | 112—118 | 0.0632562266254 | 0.200401444832 |
| C1720950881 | 22—28 | 29—35 | 0.0849303035202 | 0.196104661123 |

候选ID、完整K/V金额及全部条目见query_examples.json和query_results.jsonl；超过5条的诊断在此仅展示前5条。

### q081：C / C1005806982 / t=59 / Lc=14

集合=expansion_sensitivity_m1500；状态=FULL / CROSS / EXPANDED_ALL；候选数=1500；截止日=59。
Q=[0.0, 0.0, 0.0, 44.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84, 0.0, 0.0, 0.0, 0.0]

R=[10.565003976541536, 7.4397992271752065, 7.104697992482107, 8.731851916329044, 9.00599975447783, 8.376436022158002, 7.4522547550254545]

| 供体 | K日期 | V日期 | 距离 | 权重 |
| --- | --- | --- | --- | --- |
| C1051985937 | 36—49 | 50—56 | 0.0778236897291 | 0.00307668535548 |
| C1124718683 | 8—21 | 22—28 | 0.0780980167429 | 0.00307584145333 |
| C1152092987 | 29—42 | 43—49 | 0.0783121822884 | 0.0030751827846 |
| C1232725180 | 24—37 | 38—44 | 0.0784742921009 | 0.0030746843077 |
| C1035154937 | 29—42 | 43—49 | 0.078753794806 | 0.00307382504521 |

候选ID、完整K/V金额及全部条目见query_examples.json和query_results.jsonl；超过5条的诊断在此仅展示前5条。

## 6. 本次集合统计

以下分母均为实际调用次数，同查询不同方法各计一次；另列不同查询数。边界组NO_HISTORY计入无参考率，历史有效调用率另存sample_report.json。

| 集合 | 实验 | 调用/不同查询 | 有参考 | 扩库 | 无参考 | 有效检索率 | 扩库率 | 无参考率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boundary_diagnostic | C | 4/4 | 2 | 0 | 2 | 50.000000% | 0.000000% | 50.000000% |
| boundary_diagnostic | O | 4/4 | 2 | 0 | 2 | 50.000000% | 0.000000% | 50.000000% |
| expansion_sensitivity_m1500 | C | 1/1 | 1 | 1 | 0 | 100.000000% | 100.000000% | 0.000000% |
| expansion_sensitivity_m1500 | O | 1/1 | 1 | 1 | 0 | 100.000000% | 100.000000% | 0.000000% |
| main_sample | C | 50/26 | 48 | 0 | 2 | 96.000000% | 0.000000% | 4.000000% |
| main_sample | O | 22/14 | 20 | 0 | 2 | 90.909091% | 0.000000% | 9.090909% |

主样本合计：68/72有参考（94.444444%），扩库0%，无参考5.555556%。不能用这些小批统计声称总体性能或同簇优于全库。

## 7. 缓存、资源、保护和停止点

真实缓存位于artifacts/stage3/cache/queries（82个JSON）和cache/libraries（26个NPZ日期索引及配套JSON）。查询键含实验、客户、t、普通/人工历史、Lc、分区、方法、冻结/滚动、截止日、全部检索参数、数据/面板/配置/聚类/标签/划分/代码版本及可见Q/画像/属性hash。候选库键含路由、ell/H、stride、允许起点、实际日期上限及版本。参数或版本变化不会复用旧结果；文件损坏明确阻断。
首次82次查询缓存命中0；随后82次重放全部命中且结果一致；另用全新引擎核对主样本重载，并以测试验证日期索引的磁盘重载。单查询最多扫描45467条候选，最大距离块2048行。只分配单查询距离向量及一个block×ell键矩阵，无所有查询×候选矩阵；内存候选索引LRU上限8个。未进行全量查询耗时/峰值内存基准。
原始CSV SHA-256：e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377。基线覆盖94个前两阶段/来源文件，全部保持不变，证据见baseline.json和integrity-final.json。前两阶段DEVELOPMENT_PROGRESS原字节保存为progress-before.txt，并验证新增记录只追加。

产物：toy_result.json、requests.json、query_results.jsonl、query_examples.json、sample_report.json、acceptance-final.json、regression-data.json、regression-cluster.json、environment.lock.txt、integrity-final.json、delivery_report.md、cache/及logs/。write_delivery.py仅从实际报告生成本交付文档，未生成验收通过值。

保留解释限制：财富增长簇在现有基线中为低消费/低活跃群体，名称仅按年龄排序；未调整客户归属，也未宣称完整财富阶段画像得到验证。权重不是置信概率，R不是模型消费预测。
未执行且不在本轮范围：全量时间样本检索性能实验、真实滚动库效果实验、预测网络训练及页面开发。尚未解决的实施或本阶段验收问题：没有。
第三阶段完成后停止，等待明确指令，不自动进入第四阶段。


---

# 第四阶段4A：预测器、smoke与资源评估（2026-11-16）

4A实现和小规模验收完成。本轮授权只到smoke和资源评估；没有启动4B正式全量训练，没有进入第五阶段。M12/E08/A07的完整实验部分仍为PARTIAL。
实施顺序：规格/旧证据核对 → CPU环境 → 保存样本准入与模型/训练实现 → 紧凑缓存等价性 → O/C smoke → 分层资源测量 → 4A验收及前三阶段回归 → 追加记录。

## 1. 文件与用途

| 新增或修改文件 | 用途 |
| --- | --- |
| `src/banksim/forecast_data.py` | 保存样本的客户/分区/日期/标签/拟合截止准入，整数分标签及客户等权抽样 |
| `src/banksim/forecast_model.py` | 固定双分支MLP、float32张量、单次log1p/Softplus、参考屏蔽及有限梯度检查 |
| `src/banksim/forecast_metrics.py` | 客户等权指标、C四档等权早停分数、五种简单基线及早停状态 |
| `src/banksim/forecast_cache.py` | 按合法样本行索引的紧凑memmap缓存；双精度R/权重/距离和候选追溯、逐行校验 |
| `src/banksim/forecast_training.py` | 训练/验证、独立无检索模型、最佳轮次恢复、曲线及断点保存 |
| `src/banksim/forecast_checkpoint.py` | 模型及完整配置/尺度/中心/映射/划分存取、版本/内容不匹配拒绝 |
| `src/banksim/forecast_predict.py` | 明确SMOKE_ONLY/MODEL_NOT_TRAINED、预测追溯字段与原比例映射调用 |
| `src/banksim/forecast_resources.py` | Windows工作集及原生分配峰值测量、磁盘字节统计 |
| `src/banksim/forecast_stage.py` | smoke、benchmark、估算命令；独立显式full入口，本轮未执行 |
| `src/banksim/retrieval_version.py` | 检索依赖白名单版本；预测器新增文件不影响已有参考身份 |
| `config/retrieval_compatibility.json` | 经等价性核验的第三阶段版本兼容基线及数值依赖版本 |
| `src/banksim/retrieval_io.py` | 改为调用检索专属版本函数；加载和计算规则不变 |
| `src/banksim/retrieval_cache.py` | 允许新诊断缓存写入stage4，保留原stage3缓存 |
| `tests/test_forecast_unit.py` | 13项网络/单位/抽样/遮挡/指标/异常/版本用例 |
| `tests/test_forecast_full.py` | 9项真实准入、缓存、smoke、加载和保护证据核验 |
| `tests/conftest.py` | 新增4A范围验收及PARTIAL状态，保留前三阶段报告能力 |
| `DEVELOPMENT_PROGRESS.md` | 追加4A记录，保留前三阶段字节前缀 |

真实聚类、原始拟合标签、三对比例和命名保持不变。回归中仅运行既有合成拟合夹具；未重新拟合O/C实际聚类。训练/评价从第二阶段保存的样本表与ID名单准入，重新检查归属、时间、历史、完整标签、属性范围和拟合截止59日；C留出ID不能进入训练或供体。正式轮次按普通样本对数量补齐batch，五种模式表只提供可行候选键。

## 2. 环境与实际执行

```json
{
  "python": "3.12.14",
  "executable": "C:\\Users\\qqq\\Documents\\Codex\\2026-11-15\\banksim-x20\\outputs\\BankSim_AI_Coding_Handoff\\.venv\\Scripts\\python.exe",
  "pip": "25.0.1",
  "torch": "2.15.0+cpu",
  "cuda_build": null,
  "cuda_available": false,
  "logical_cpus": 18,
  "run_torch_threads": 2,
  "run_blas_threads": 2,
  "existing_dependency_changes": {}
}
```

CPU PyTorch安装依据：https://pytorch.org/get-started/locally/ 。实际下载torch-2.15.0+cpu Windows cp312 wheel（日志显示126.6MB）；没有安装CUDA、torchvision或torchaudio。只修改项目.venv。安装前所有已存在依赖逐项比较均未改变，pip check无损坏依赖。

```text
cloudpickle==3.1.2
colorama==0.4.6
filelock==4.2.4
fsspec==2026.9.0
iniconfig==2.3.1
Jinja2==3.1.6
joblib==1.6.0
MarkupSafe==3.0.3
mpmath==1.3.0
narwhals==2.28.0
networkx==3.7
numpy==2.5.4
packaging==26.3
pandas==3.1.1
pluggy==1.7.0
pyarrow==26.0.1
Pygments==2.21.0
pytest==9.2.0
python-dateutil==4.0.0
scikit-learn==1.9.1
scipy==1.18.1
setuptools==78.1.0
sympy==1.14.0
threadpoolctl==3.8.0
torch==2.15.0+cpu
typing_extensions==4.16.0
tzdata==2026.6
```

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu --constraint artifacts/stage4/environment-before.lock.txt --report artifacts/stage4/logs/install-torch-authorized.json
```
结果/日志：仅CPU包及必需依赖；已有包未变

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m banksim.forecast_stage --mode smoke
```
结果/日志：logs/smoke-accepted.log

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m banksim.forecast_stage --mode benchmark
```
结果/日志：logs/benchmark.log

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m banksim.forecast_stage --mode estimate
```
结果/日志：logs/resource-estimate.log

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_forecast_unit.py tests/test_forecast_full.py -q --basetemp artifacts/stage4/pytest-accepted -o cache_dir=artifacts/stage4/pytest-cache-accepted --acceptance-stage=4 --acceptance-report artifacts/stage4/acceptance-4a-final.json
```
结果/日志：logs/acceptance-4a-final.log

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_visibility.py -q --basetemp artifacts/stage4/pytest-data -o cache_dir=artifacts/stage4/pytest-cache-data --acceptance-stage=1 --acceptance-report artifacts/stage4/regression-data.json
```
结果/日志：logs/regression-data.log

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stage2_unit.py tests/test_stage2_full.py -q --basetemp artifacts/stage4/pytest-cluster -o cache_dir=artifacts/stage4/pytest-cache-cluster --acceptance-stage=2 --acceptance-report artifacts/stage4/regression-cluster.json
```
结果/日志：logs/regression-cluster.log

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_retrieval_full.py -q --basetemp artifacts/stage4/pytest-retrieval -o cache_dir=artifacts/stage4/pytest-cache-retrieval --acceptance-stage=3 --acceptance-report artifacts/stage4/regression-retrieval.json
```
结果/日志：logs/regression-retrieval.log

上述命令实际执行时通过 `2>&1 | Tee-Object artifacts/stage4/logs/<对应日志>` 保存控制台输出。另执行了pip freeze、pip check、compileall及解释器/设备检查。

首次沙箱pip调用无法找到torch并报告临时目录权限问题，联网授权后CPU安装成功；失败日志仍保留。pytest沿用此前验证的项目内授权运行方式，避免沙箱临时目录WinError 5。
开发中补强了整数分标签直接读取、完整checkpoint配置/网络结构/划分存档和版本校验；前两次成功smoke分别保留在smoke_initial与smoke_metadata_draft，当前验收模型只取smoke/。初次4A集成验收21通过/1失败：Python与NumPy对未舍入日预测求和相差约1e-14，测试改为绝对1e-12、相对0容差后22项通过；未改变预测值或美化指标。原acceptance-4a.json及日志保留，最终为acceptance-4a-final.json。

| 程序 | 实际总耗时（秒） | 加载及准入（秒） |
| --- | --- | --- |
| smoke | 36.649175 | 27.385174 |
| benchmark | 73.947010 | 30.971707 |
| estimate | 119.856864 | 59.904902 |

## 3. O/C smoke实际结果

每实验取4名具有59/62/89合法普通窗口的训练客户，共12个普通(customer,t)对、60行普通+冷窗模式。smoke batch=16，每轮抽16条，3轮共48条；种子42。验证取4名合法客户、t=119/126：O普通8条，C四个Lc共32条。客户选择、原始source_row和完整窗口在各run的train_selection.parquet及validation_selection.parquet中。正式默认batch256/epochs100没有修改。
训练/标签尺度a：O=26.81、C=26.88，直接读取冻结产物。以下训练loss为scaled MSE，验证为原金额单位客户等权MSE；C先各Lc客户等权再四档等权。这些小样本指标仅检查流程，不证明模型优于基线。

| 实验/模型 | 训练客户 | 每轮抽样/轮次 | 各轮训练loss | 各轮验证MSE | best轮 | best验证MSE |
| --- | --- | --- | --- | --- | --- | --- |
| O/retrieval | 4 | 16/3 | 4.723828, 7.271336, 5.298706 | 3793.408410, 3796.963648, 3800.462249 | 1 | 3793.408410 |
| O/no_retrieval | 4 | 16/3 | 4.719621, 7.285642, 5.332339 | 3766.481090, 3768.608938, 3771.294337 | 1 | 3766.481090 |
| C/retrieval | 4 | 16/3 | 4.698138, 7.231236, 5.272568 | 1164.069028, 1164.155642, 1163.677483 | 3 | 1163.677483 |
| C/no_retrieval | 4 | 16/3 | 4.693401, 7.246132, 5.303114 | 1165.631575, 1164.664668, 1163.217339 | 3 | 1163.217339 |

| 实验/模型 | best验证RMSE | MAE | TotalAE |
| --- | --- | --- | --- |
| O/retrieval | 61.590652 | 25.230394 | 128.875186 |
| O/no_retrieval | 61.371664 | 25.138728 | 129.440958 |
| C/retrieval | 34.112233 | 18.519546 | 70.157689 |
| C/no_retrieval | 34.105458 | 18.582394 | 70.571044 |

C汇总RMSE为四档RMSE的算术平均，每档内部先聚合客户MSE再开根号；没有平均客户RMSE。每档完整指标和五基线结果均在summary.json，方法共用相同客户/t/Lc/标签。no_retrieval是独立初始化、优化器和训练产物，非有检索模型临时关参考。

## 4. 存取、恢复与错误状态

固定网络实际参数数=10599。输入q_log/q_mask为B×28、r_log为B×7、r_mask为B×1，标签B×7，均float32。历史左补齐；真实0保持mask=1。参考使用现成r_log，不二次log。输出乘a，无expm1或强制截0。
四个smoke模型全部重载预测一致、best恢复一致。O的最佳轮为1而last为3，实际恢复最佳权重；C最佳轮为3。同分与连续10轮不改善由独立用例验证，本次3轮smoke没有声称触发耐心早停。
best.pt/last.pt保存网络、Adam状态、epoch/best/bad_epochs、PyTorch随机状态、参考遮挡随机状态和客户抽样生成器状态；JSON另存完整默认配置、smoke实际覆盖参数、a、变换、中心、group_map、实际split IDs、包版本、源hash、检索版本、模型代码版本及样本清单hash。支持重载检查；本轮未测试跨进程断点续训流程。
无权重返回MODEL_NOT_TRAINED且预测为null；smoke要求显式kind='smoke'并输出SMOKE_ONLY；以full方式载入smoke、或改H/L、m、a、数据/聚类/检索版本均拒绝CONFIG_MISMATCH。非有限loss/gradient明确TRAINING_FAILED；C任一验证档为空在训练开始前VALIDATION_EMPTY。

## 5. 检索、缓存、内存及训练速度

初始测量：default_rng(42)，O/C训练的SELF及Lc=1/3/7/14跨客户路径、O普通验证和C四档验证，15层各无放回取12条，共180条。全部与第三阶段直接结果逐项核对候选ID/日期/排序/距离/权重/R一致。数据准备、直接首次计算、紧凑缓存首次写入/命中、原JSON首次写入/命中分开计时。compact_first在直接查询之后测量，可能受已热日期索引影响；全量缓存填充估算采用direct retrieval而未用这个较快值代替。

| 分层 | n | 数据准备ms | 直接计算ms | 紧凑首次ms | 紧凑命中ms | JSON首次ms | JSON命中ms | batch256准备ms | batch256训练ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O_train_SELF_Lc0 | 12 | 0.999 | 10.125 | 9.739 | 0.143 | 33.904 | 25.280 | 47.923 | 2.868 |
| O_train_CROSS_Lc1 | 12 | 0.894 | 79.926 | 44.651 | 0.140 | 119.685 | 23.955 | 41.362 | 1.768 |
| O_train_CROSS_Lc3 | 12 | 0.861 | 84.820 | 43.333 | 0.251 | 131.147 | 27.041 | 56.733 | 1.820 |
| O_train_CROSS_Lc7 | 12 | 1.207 | 84.015 | 46.377 | 0.303 | 128.636 | 28.421 | 53.616 | 1.938 |
| O_train_CROSS_Lc14 | 12 | 1.140 | 72.598 | 39.139 | 0.204 | 91.487 | 26.502 | 43.731 | 2.228 |
| O_validation_SELF_Lc0 | 12 | 0.681 | 7.236 | 6.964 | 0.127 | 26.585 | 22.841 | 51.724 | 未执行（验证分区禁止训练） |
| C_train_SELF_Lc0 | 12 | 0.704 | 8.869 | 7.902 | 0.182 | 30.083 | 28.048 | 53.574 | 1.890 |
| C_train_CROSS_Lc1 | 12 | 1.074 | 57.951 | 30.250 | 0.293 | 88.567 | 23.334 | 43.839 | 2.795 |
| C_train_CROSS_Lc3 | 12 | 0.974 | 61.668 | 33.041 | 0.390 | 105.883 | 26.779 | 54.166 | 2.619 |
| C_train_CROSS_Lc7 | 12 | 1.019 | 55.828 | 33.115 | 0.142 | 85.614 | 29.271 | 40.629 | 1.660 |
| C_train_CROSS_Lc14 | 12 | 0.926 | 55.946 | 32.935 | 0.248 | 98.764 | 23.896 | 69.833 | 2.255 |
| C_validation_CROSS_Lc1 | 12 | 1.162 | 46.771 | 41.970 | 0.143 | 47.859 | 26.348 | 49.010 | 未执行（验证分区禁止训练） |
| C_validation_CROSS_Lc3 | 12 | 1.023 | 44.093 | 43.581 | 0.337 | 47.072 | 24.690 | 51.554 | 未执行（验证分区禁止训练） |
| C_validation_CROSS_Lc7 | 12 | 0.930 | 50.400 | 43.810 | 0.155 | 52.846 | 28.448 | 45.950 | 未执行（验证分区禁止训练） |
| C_validation_CROSS_Lc14 | 12 | 0.785 | 48.718 | 44.463 | 0.166 | 51.645 | 28.343 | 59.439 | 未执行（验证分区禁止训练） |

训练速度测量仅在训练分区进行：12条已准入样本重复组为256，1步预热后测10步，不保存为模型效果。验证分区从未做梯度更新。随后另取O512条、C每Lc128条（合计512条）验证样本，仅做前向与评价；warm为3次均值，降低极小批固定开销对外推的影响。样本清单为benchmark/validation_timing_O.parquet与C.parquet。

| 实验 | 验证样本/客户 | 首次总秒 | 缓存命中总秒（三次均值） | 三次实测秒 |
| --- | --- | --- | --- | --- |
| O | 512/479 | 9.444031 | 0.743726 | [0.46229870000388473, 0.6544194000016432, 1.114458399999421] |
| C | 512/352 | 46.348345 | 0.304743 | [0.26904929999727756, 0.29133330000331625, 0.35384609998436645] |

实际峰值工作集=1,634,668,544字节（1.522GiB），使用Windows GetProcessMemoryInfo的进程生命周期PeakWorkingSetSize，包含NumPy/PyTorch原生分配及加载完整合法样本表；不是只统计Python堆。smoke峰值=1,404,260,352字节。正式训练峰值未实测。
紧凑缓存每行306字节（R和r_log为float64，保留5条供体/键末日/距离/权重/library版本与校验码）。180条分层测量对应15个独立catalog，实际紧凑目录共1,290,880字节，含重复固定供体字典/元数据；JSON查询文件共1,772,293字节，日期库文件另15,914,794字节。小批文件固定开销不能直接当百万级每行成本，故全量另按dtype字节数估算。
缓存键含实验、分区、保存样本表hash/行索引、数据/聚类/检索语义版本、全部检索参数和方法。R/距离/权重以float64保存；进入网络时才转float32。真正检索依赖代码或数值包版本变化使缓存失效；新增无关预测器文件不会改变旧检索版本。第三阶段原缓存未改动，旧查询重载回归通过。

## 6. 种子42正式训练规模与资源估算（未执行）

以下只估算每实验一个有检索、same_group方法、CPU2线程模型；使用正式batch256、最多100轮。普通模式样本数计算每轮抽样量；五模式总行数只用于缓存容量上界。时间不是实测全量结果或承诺上限。

| 实验 | 有效训练客户 | 普通合法对 | 每轮抽样 | batch/轮 | 验证样本 | 缓存行上界 | 缓存负载GB | 热缓存轮秒 | 首轮全未命中情景小时 | 一次填充+100轮小时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O | 4111 | 221680 | 221696 | 866 | 98664 | 1206382 | 0.369153 | 187.307 | 2.984 | 25.809 |
| C | 2877 | 155136 | 155136 | 606 | 59136 | 834329 | 0.255305 | 68.502 | 2.218 | 13.034 |

O/C两模型此100轮情景合计约38.84小时。假设所有合法参考最多填充一次，之后命中；训练按普通0.5、每Lc0.125外推，忽略少量可行长度差异；短普通窗填充耗时用跨客户均值近似；验证按512条实测线性外推。CPU调度、日期索引LRU命中、磁盘页缓存、客户分布与早停会显著改变实耗；验证三次计时已有可见波动。
两实验训练+验证缓存负载合计624,457,566字节（0.624GB/0.582GiB）。原JSON方案仅查询文件外推约20.24GB、2,040,711个查询文件，尚不含日期索引、文件系统开销。
磁盘规划分项：源训练/验证Parquet合计3,479,961字节；新增source_row/lc_key两列按未压缩16字节/行估算32,651,376字节；已测4个smoke模型的best/last权重合计1,175,208字节。与缓存负载合计的规划小计约0.662GB，另需元数据、日志、文件系统和版本余量；可预留2GB用于这一对正式模型的新产物。这里不包含既有数据/诊断缓存/环境，亦不包含多种子及额外方法副本。正式内存峰值尚未实测；最多约0.62GB紧凑缓存映射页在访问中可能额外进入工作集，不能将当前smoke峰值当正式内存上限。

## 7. 验收、回归及未完成项

| 项目 | 实际状态 | 范围和证据 |
| --- | --- | --- |
| M01 | PASS | 左补齐；真实0掩码1、补齐掩码0；Q只log1p一次 |
| M02 | PASS | 固定形状/float32、56→64/8→32/96→64→7、全部bias、一次Softplus；实际前后向 |
| M03 | PASS | 读取冻结a=26.81/26.88；日总额中位数、0—59训练ID、无正值fallback及未来/留出不影响夹具 |
| M04 | PASS | 整数分/100/a标签；预测×a；已知scaled MSE=0.5；没有expm1 |
| M05 | PASS | 无参考及全0历史仍用相同网络；预测有限非负，不人为截0 |
| M06 | PASS | 先客户均匀再合法t均匀；50%截短及可行Lc均匀；每轮普通对数量向上补齐batch |
| M07 | PASS | 训练20%同步遮挡r_log/r_mask、不改标签；验证不消耗遮挡随机数 |
| M08 | PASS | O/C独立无检索smoke确实重新训练、模型标识和权重不同；同一验证键 |
| M09 | PASS | 合成序列验证同分保留早轮和10轮耐心；真实O第1轮best从第3轮last恢复 |
| M10 | PASS | 非有限loss/gradient明确TRAINING_FAILED；缺失C档明确VALIDATION_EMPTY并在训练前阻断 |
| M11 | PASS | O/C保存重载预测一致；缺权重MODEL_NOT_TRAINED；smoke不能冒充full |
| M12 | PARTIAL | CPU、42/43/44初始化合同及seed42 smoke已验证；正式三种子训练未执行 |
| E05 | PASS | 不等客户窗口数夹具MSE=19、RMSE=sqrt(19)、MAE=3.5 |
| E06 | PASS | [18,28]对[20,30]指标4/2/2/4；真实预测总额使用未舍入数值 |
| E07 | PASS | 每Lc客户内平均→客户等权→四档等权；任意空档失败 |
| E08 | PARTIAL | 五基线与独立无检索smoke共用同一验证键/标签；正式全库/同簇等完整对照未执行 |
| E12 | PASS | 配置、H/L、m、a、数据、聚类、检索和模型代码/包版本及文件内容校验；缓存失效与旧缓存保留 |
| A07 | PARTIAL | smoke推理字段完整且明确标注；正式模型尚未训练，正式prediction验收未完成 |

最终4A测试22/22通过，但不是整个第四阶段全部验收完成。M12正式42/43/44实验、E08正式对照矩阵、A07正式模型prediction仍为部分完成；本轮范围内没有未解决的实现失败。

data回归：54 passed in 4.63s；D01=PASS、D02=PASS、D03=PASS、D04=PASS、D05=PASS、D06=PASS、D07=PASS、D08=PASS、D09=PASS、D10=PASS、D11=PASS、D12=PASS、D13=PASS、D14=PASS、D15=PASS。
cluster回归：46 passed in 8.62s；C01=PASS、C02=PASS、C03=PASS、C04=PASS、C05=PASS、C06=PASS、C07=PASS、C08=PASS、C09=PASS、C10=PASS、C11=PASS、C12=PASS、C13=PASS、C14=PASS、E01=PASS、E02=PASS、E03=PASS、E04=PASS、A01=PASS、A02=PASS、A03=PASS、A04=PASS、A05=PASS、A06=PASS。
retrieval回归：34 passed in 11.47s；R01=PASS、R02=PASS、R03=PASS、R04=PASS、R05=PASS、R06=PASS、R07=PASS、R08=PASS、R09=PASS、R10=PASS、R11=PASS、R12=PASS、R13=PASS、R14=PASS、R15=PASS。

257个受保护文件实际校验数=257，全部SHA-256不变。源码中的candidates/retrieval/数据/画像/分群/划分/比例计算也与source-before记录一致。DEVELOPMENT_PROGRESS只追加，旧字节前缀另存progress-before.txt并校验。
尚未执行：正式全量训练、正式测试集效果、多种子训练、完整方法对照、正式模型推理验收、正式全量内存实测及跨进程断点续训试验。未开发界面、预算、收入预测或资产收益优化；客群名称仍仅按年龄排序，未声称完整财富阶段画像已被验证。

## 8. 产物与下一步命令

当前可验收模型：artifacts/stage4/smoke/{O,C}/{retrieval,no_retrieval}/，每目录包含best/last.pt及JSON、config.json、train_selection.parquet、validation_selection.parquet、curve.json、sampling_trace.json、summary.json、prediction_example.json。初轮目录smoke_initial和smoke_metadata_draft只保留开发历史，不作为当前模型。
资源及缓存：benchmark/report.json、validation_timing_*.parquet、compact/、diagnostic_json/、validation_512_compact/；训练缓存smoke/cache/。验收：acceptance-4a-final.json及三份regression-*.json。环境：environment-before.lock.txt、environment.lock.txt、environment.json。完整性：baseline.json、source-before.json、integrity-final.json。执行日志全部位于logs/。
以下命令仅供下一次明确授权4B后执行，本轮未执行；没有artifacts/stage4/full目录。

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m banksim.forecast_stage --mode full --experiment both --seed 42 --variant retrieval --method same_group --confirm-full-training
```

正式产物将写入artifacts/stage4/full/{O,C}/retrieval_same_group_seed42/。CLI默认不会自动运行full；必须独立指定--mode full及--confirm-full-training。独立无检索训练可用--variant no_retrieval，完整方法矩阵另待授权。
4A结束，等待明确下一步指令，不自动开始4B或第五阶段。


Hardware capacity snapshot after 4A (not a full-training measurement): physical RAM total 31.435 GiB, available 12.300 GiB; project-volume free space 271.476 GiB. Details: artifacts/stage4/hardware_snapshot.json. Full training remains unstarted.


# 4A补强：断点续训、可靠保存和训练进度（2026-11-16）

本次授权范围已完成。仅运行小规模 smoke 和测试；未启动4B，未读取测试标签做调参或效果选择。
算法依据为 ALGORITHM_SPEC.md；默认参数、网络、Adam、抽样、参考遮挡、早停、检索、聚类和比例规则均保持原协议。

## 1. 文件和实现

| 文件 | 用途 |
| --- | --- |
| src/banksim/forecast_runner.py（新增） | 批次级训练状态机，恢复last、模型/Adam/三个随机状态、轮内累计值、曲线和早停状态 |
| src/banksim/forecast_runtime.py（新增） | OS进程写锁、Windows长路径访问、独立线程进度日志 |
| src/banksim/forecast_train_cli.py（新增） | 明确区分new、resume、infer-best；正式训练必须单独确认；smoke不能升级为full |
| src/banksim/forecast_checkpoint.py（修改） | 不可变分代权重及元数据、SHA-256、原子提交清单、上一提交回退与保留/清理 |
| src/banksim/forecast_cache.py（修改） | 缓存写锁、payload先写/ready最后提交、校验、关闭及故障注入接口；缓存语义和版本键不变 |
| src/banksim/forecast_training.py（修改） | 保留训练配置和评价函数，训练入口委托统一续训状态机 |
| tests/test_forecast_resume.py、test_forecast_resume_full.py、resume_fault_worker.py（新增） | 实际子进程崩溃、锁冲突、缓存恢复、配置/版本拒绝及跨进程结果核验 |
| tests/test_forecast_full.py、conftest.py（修改） | 显式关闭缓存后重开；支持指向新smoke产物，保留旧报告 |
| artifacts/stage4/resume_validation/ | 新证据、运行脚本、模型和缓存；DEVELOPMENT_PROGRESS.md仅追加本记录 |

恢复状态包括 epoch、next_batch、completed_steps、phase、loss_sum/sample_count、遮挡计数、准备/训练耗时、best_score/best_epoch/bad_epochs、模型/Adam、PyTorch/客户抽样/参考遮挡随机状态及既有曲线和smoke抽样追踪。完成优化步骤后才移动游标和提交。第一批前也有零步断点。
验证前提交phase=validating；中断后整轮验证重跑。验证结果、早停更新和曲线追加在同一次清单提交生效，不重复更新。best仅用于推理/最终评价，resume明确读取last。
checkpoint_manifest.json是提交点；checkpoint_previous.json保存上一完整提交。checkpoints/里的未提交文件和临时文件永远不算有效断点。校验当前提交失败时使用上一完整提交；配置/代码不兼容则拒绝，不靠回退伪造兼容。正常清理仅保留当前和上一提交引用的last/best分代，避免正式训练随批次数无限增长。
run.lock与每份缓存的writer.lock使用内核互斥锁；退出自动释放，不依据PID或锁文件年龄抢锁，不删除锁文件。第二个写进程明确RUN_LOCKED/CACHE_LOCKED并输出writer_conflict日志。
progress.jsonl及标准输出默认每30秒输出实验、变体、epoch、批次、已用时间、实时缓存命中/未命中和最近有效checkpoint。整轮另输出训练MSE与验证MSE。本次没有输出未经测量的ETA。

## 2. 环境及实际命令

项目.venv：Python 3.12.14，PyTorch 2.15.0+cpu，CPU，训练PyTorch/BLAS均2线程；无新增、升级依赖。完整版本在environment.lock.txt，与原4A快照一致。

独立进程对照驱动命令（实际执行，12个子进程的完整命令、PID、退出码与耗时见final/commands.json）：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe artifacts/stage4/resume_validation/run_comparison.py artifacts/stage4/resume_validation/final
```

实际执行的新训练/续训命令示例；B的总计划始终是3轮，仅测试停止点不同。下列目录现已完成，新训练重做时必须使用新的run-dir。

```powershell
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli new --run-dir artifacts/stage4/resume_validation/final/comparison/O/B --experiment O --kind smoke --seed 42 --variant retrieval --method same_group --smoke-profile resume-validation --checkpoint-every-batches 1 --stop-after-steps 2
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli resume --run-dir artifacts/stage4/resume_validation/final/comparison/O/B --experiment O --kind smoke --seed 42 --variant retrieval --method same_group --smoke-profile resume-validation --checkpoint-every-batches 1
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli infer-best --run-dir artifacts/stage4/resume_validation/final/comparison/O/B --experiment O --kind smoke
```

infer-best命令实际在保存的验证名单首个合法样本上调用best推理；输出SMOKE_ONLY，见infer-best.log。通用客户接口仍为forecast_predict.predict_customer。

## 3. 跨进程对照结果

选择：各实验取按ID排序、具有59—70全部合法普通切点的前4名训练客户；48个普通(customer,t)对，240条保存的普通/冷窗可行模式记录；batch=16，每轮48次抽样/3批，总3轮9批144次抽样。验证取119/126日的前4名合法客户，O为8条，C四档Lc为32条。只使用保存的训练/验证名单。
A连续完成3轮；B在第1轮第2批后退出（32条已完成样本，尚无best），新Python恢复；C第1轮结束后退出，新Python恢复；另加O/D在第3轮验证更新早停/曲线但尚未提交时os._exit(75)，新进程恢复。
容差在运行前写入final/comparison-plan.json：绝对0、相对0。比较张量用torch.equal，其他数值精确相等；仅排除耗时、PID、分代名称和缓存命中统计。
模型参数、Adam状态、后续抽样及参考遮挡、PyTorch/采样器/遮挡随机状态、轮内累计损失、曲线、验证指标、best、早停及最终预测全部一致。

| 实验 | 比较 | 训练/验证客户 | 每轮抽样/批次 | best轮 | 最终bad_epochs | 最大数值差 |
| --- | --- | --- | --- | --- | --- | --- |
| O | A vs B/C/D | 4 / 4 | 48 / 3 | 1 | 2 | 0 |
| C | A vs B/C | 4 / 4 | 48 / 3 | 3 | 0 | 0 |

| 实验/轮 | 归一化训练MSE | 原金额客户等权验证MSE |
| --- | --- | --- |
| O/1 | 3.608263174693 | 3809.488627477480 |
| O/2 | 3.616433223089 | 3821.510885057880 |
| O/3 | 3.664514064789 | 3833.824413242713 |
| C/1 | 3.594041109085 | 1149.942306180580 |
| C/2 | 3.595134059588 | 1139.321050571501 |
| C/3 | 3.644392331441 | 1133.516315935098 |

12个对照子进程实测耗时合计369.393秒（含各自Python、数据与模型加载）；逐次耗时在commands.json，不能外推为正式全量训练时间。
O验证中断前已提交bad_epochs=1；恢复后为2，曲线仍恰好3条。证据：final/validation-interrupted-state.json及final/comparison.json。零差异证明限于本CPU/依赖/线程环境中的这份小样本，不代表全量效果或优于基线。

## 4. 故障、缓存及不兼容处理

| 检查 | 实际结果 |
| --- | --- |
| 权重写完、JSON写完、清单提交前进程退出 | 回到上一完整提交，未提交权重/JSON不被当成断点 |
| 清单提交后进程退出 | 读取新完整提交 |
| 最新提交权重损坏 | 哈希检测并回到上一有效提交 |
| 无提交清单但存在未提交文件 | MODEL_NOT_TRAINED |
| 第一轮中途无best恢复 | O/C均从last继续，完成结果与连续训练相同 |
| 缓存payload写完、ready未提交退出 | 已完整行命中，未完成行重新计算 |
| 缓存ready提交后退出 | 完整行继续命中 |
| ready=1但行校验失败 | 明确CONFIG_MISMATCH，不把坏值送入网络 |
| 两进程写同一运行目录或同一缓存 | RUN_LOCKED/CACHE_LOCKED，冲突输出日志；进程死亡后锁可重新取得 |
| seed/变体/方法/轮数/batch/实验/样本清单/a/数据/聚类/检索/代码不匹配 | 明确CONFIG_MISMATCH，未更新checkpoint提交清单 |
| SMOKE_ONLY升为full、原4A旧代码模型加载 | 拒绝；原模型与元数据原样保留 |
| 已存在运行目录调用new | 拒绝并要求resume或新目录 |
| 轮内进度 | 独立线程在模拟长批次期间持续输出所需字段，间隔上限60秒 |

故障注入采用独立Python进程os._exit，未模拟硬件断电。存储故障测试使用确定性夹具；实际训练恢复及原检索等价性另在真实合法样本上验证。

开发失败证据保留：第一次pytest受沙箱WinError5阻挡；第二次10项中6项因Windows长路径失败、4项通过，见tests-resume-second.xml；长路径修复后10项全通过。首次对照驱动比较Adam整数键时发生脚本AttributeError，未得出数值不一致结论；原comparison/与logs/保留。最终驱动修正后用新代码在final/全部重跑并通过，不修改旧模型元数据。

## 5. 验收与回归

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| resume-third | 10通过，0失败/错误/跳过；109.793秒 | tests-resume-third.xml |
| resume-full | 11通过，0失败/错误/跳过；18.147秒 | tests-resume-full.xml |
| data回归 | 54 passed in 4.93s | regression-data.json/.log |
| cluster回归 | 46 passed in 9.47s | regression-cluster.json/.log |
| retrieval回归 | 34 passed in 14.09s | regression-retrieval.json/.log |
| forecast回归 | 22 passed in 44.82s | regression-forecast.json/.log |

第一阶段D01—D15、第二阶段C01—C14/E01—E04/A01—A06、第三阶段R01—R15全部通过。4A的M01—M11/E05—E07/E12通过；M12/E08/A07继续为PARTIAL，正式多种子、完整方法比较和正式模型预测未执行，不能用smoke宣称第四阶段全部通过。
新的4A regression/smoke/{O,C}/{retrieval,no_retrieval}共4个独立smoke模型全部重跑、保存/重载验收通过；各轮损失、验证MSE、best和遮挡计数与原4A相同，原4A模型保留且由当前代码明确拒绝版本不兼容。
受保护的原始数据、配置/规格、前三阶段与原4A产物共884个文件逐一SHA-256检查，全部未变。算法模块哈希未变，检索语义版本和既有缓存键未变。DEVELOPMENT_PROGRESS.md保留原字节前缀，只追加。证据integrity.json。

实际回归/测试命令（均在项目.venv，全部新报告写入本目录；pytest因已确认的Windows沙箱临时目录问题使用授权执行）：

```powershell
.\.venv\Scripts\python.exe -m banksim.forecast_stage --mode smoke --experiment both --output artifacts/stage4/resume_validation/regression
.\.venv\Scripts\python.exe -m pytest tests/test_forecast_resume.py -q --basetemp artifacts/stage4/resume_validation/pytest-resume-third -o cache_dir=artifacts/stage4/resume_validation/pytest-cache-third --junitxml artifacts/stage4/resume_validation/tests-resume-third.xml
.\.venv\Scripts\python.exe -m pytest tests/test_forecast_resume_full.py -q --basetemp artifacts/stage4/resume_validation/t-resume-full -o cache_dir=artifacts/stage4/resume_validation/c-resume-full --junitxml artifacts/stage4/resume_validation/tests-resume-full.xml
.\.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_visibility.py -q --basetemp artifacts/stage4/resume_validation/t-data -o cache_dir=artifacts/stage4/resume_validation/c-data --acceptance-stage 1 --acceptance-report artifacts/stage4/resume_validation/regression-data.json --forecast-artifacts-root artifacts/stage4/resume_validation/regression
.\.venv\Scripts\python.exe -m pytest tests/test_stage2_unit.py tests/test_stage2_full.py -q --basetemp artifacts/stage4/resume_validation/t-cluster -o cache_dir=artifacts/stage4/resume_validation/c-cluster --acceptance-stage 2 --acceptance-report artifacts/stage4/resume_validation/regression-cluster.json --forecast-artifacts-root artifacts/stage4/resume_validation/regression
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_retrieval_full.py -q --basetemp artifacts/stage4/resume_validation/t-retrieval -o cache_dir=artifacts/stage4/resume_validation/c-retrieval --acceptance-stage 3 --acceptance-report artifacts/stage4/resume_validation/regression-retrieval.json --forecast-artifacts-root artifacts/stage4/resume_validation/regression
.\.venv\Scripts\python.exe -m pytest tests/test_forecast_unit.py tests/test_forecast_full.py -q --basetemp artifacts/stage4/resume_validation/t-forecast -o cache_dir=artifacts/stage4/resume_validation/c-forecast --acceptance-stage 4 --acceptance-report artifacts/stage4/resume_validation/regression-forecast.json --forecast-artifacts-root artifacts/stage4/resume_validation/regression
```

## 6. 产物及未解决问题

所有新增验证产物在artifacts/stage4/resume_validation/。最终跨进程权重、分代清单、清单样本、缓存、曲线和逐次日志在final/comparison/{O,C}/{A,B,C}，另有O/D；各运行含checkpoints/、checkpoint_manifest.json、checkpoint_previous.json、config.json、train_selection.parquet、validation_selection.parquet、summary.json、sampling_trace.json、progress.jsonl。
逐项新测试见tests-resume-third.xml和tests-resume-full.xml；阶段回归见regression-*.json；汇总见delivery_summary.json；原证据保护见baseline.json/integrity.json；运行前方案和零容差见final/comparison-plan.json。旧模型、旧报告及失败证据均保留。
本次补强范围内尚未解决问题：没有。未执行：4B正式全量训练、正式测试集效果、多种子完整实验、GPU训练。4A smoke不能作为正式模型使用。

## 7. 下一步O实验seed42正式训练命令（仅展示，未执行）

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli new --run-dir artifacts/stage4/full/O/retrieval_same_group_seed42 --experiment O --kind full --seed 42 --variant retrieval --method same_group --checkpoint-every-batches 25 --progress-seconds 30 --confirm-full-training
```

正式协议仍为batch256、最多100轮、patience10。以后若恢复该正式运行，将new改为resume，保持run-dir、实验、kind、seed、变体和方法一致；仍需要单独明确授权。该命令只启动O，不自动启动C或第五阶段。

本次4A补强结束，等待下一步明确指令。

补充核对：命令行best推理另以UTF-8直接捕获，避免PowerShell管道中文编码影响日志；原infer-best.log保留。完整正确记录为artifacts/stage4/resume_validation/prediction_example.json、infer-best-utf8.log，执行命令及耗时为inference-command.json。实际返回SMOKE_ONLY，客户C1000148617、t=119，7日总额138.70789667844772；客群为财富安享客户，原固定比例0.85/0.15保持不变。

## 2026-11-17 4B-O正式训练启动记录（进行中，非完成报告）

按本轮明确授权启动O/full/seed42/retrieval/same_group，batch256、max_epochs100、patience10，每25批checkpoint、每30秒进度；未启动C、其他种子、其他模型或测试集评价。
启动前实际核验：项目.venv Python3.12.14/PyTorch2.15.0+cpu，42个已验收源码文件及884个受保护文件哈希一致，依赖版本一致；运行目录无既有训练，运行/缓存锁探测通过，无活动Python训练进程。可用物理内存18035830784字节，磁盘空闲291419205632字节。
完整O训练客户4111名、普通合法切点对221680、可用历史模式记录1107718；每轮221696次抽样/866批。完整验证4111名客户、98664条样本，预热金额尺度a=26.81。
正式启动时间北京时间2026-11-17 17:37:50；Windows虚拟环境启动器PID30092，实际Python训练PID9124，监督PID29932；另设实际进程内存监测及完成后best完整验证/报告监督器。确切命令、环境和配置见artifacts/stage4/full_operations/O_retrieval_same_group_seed42/launch.json及preflight.json。所有src、config、冻结模型和依赖保持不变。
运行目录artifacts/stage4/full/O/retrieval_same_group_seed42/，训练标准输出/错误、进程退出信息及外部内存监测位于artifacts/stage4/full_operations/O_retrieval_same_group_seed42/。截至本启动记录，仅第一轮训练进行中，尚无完整验证结果，不能宣称训练完成。最终停止、指标、曲线和预测检查由后续完成报告追加。

监测恢复记录：18:39左右，会话工具发生gRPC等待超时及exec运行时替换；随后重新检查原PID9124仍在训练，第225批断点有效，没有另起训练或调用resume。详见full_operations/O_retrieval_same_group_seed42/monitoring_disruptions.json。
外部监督修正：用项目Python --version短进程实测确认PowerShell Process.ExitCode可能为null（exit-code-probe.json），因此增加持有Windows进程句柄的exit_receipt.py，使用WaitForSingleObject/GetExitCodeProcess/GetProcessTimes记录真实退出码和精确时间。只替换完成监督器PID29908为31764；正式训练PID9124及启动器30092未动。训练代码、依赖、配置和实验协议没有变化。


# 4B-O运行状态（未完成报告）

核验时间：2026-11-17T19:34:42.488285+08:00。实际状态：RUNNING_NOT_COMPLETE。

已实际运行正式O/full/seed42/retrieval/same_group，原Python训练PID 9124存活；当前第1轮，已完成388/866批，完整完成轮数为0。尚未完成第一轮验证，无best模型，无最终模型/五基线指标或正式曲线。

最近有效断点：epoch=1，batch=375，分代=f57712a7612c4f829f66dc134380ae3c。清单、权重/元数据SHA及完整样本清单SHA已再次核验。未重启训练、未调用resume。

完整规模：训练客户4111名、普通合法切点对221680、可用历史模式记录1107718；每轮221696次抽样/866批，完整验证4111名客户/98664条。batch256、max_epochs100、patience10，每25批保存、每30秒进度。代码、依赖、冻结数据/预处理/聚类和规则未修改。

实际命令：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli new --run-dir artifacts/stage4/full/O/retrieval_same_group_seed42 --experiment O --kind full --seed 42 --variant retrieval --method same_group --checkpoint-every-batches 25 --progress-seconds 30 --confirm-full-training
```

最近心跳训练阶段计时为6941.062秒，仍在增长，不能当作完整训练耗时。缓存命中/未命中=6572/92837；实际训练进程观测到的生命周期峰值工作集=1569357824字节（30秒采样读取PeakWorkingSet64，包含原生分配，最终未采样间隔可能遗漏更高峰值）。运行目录现有377857747字节，缓存369318175字节；均为状态报告写入前快照。缓存已分配空间不等于所有行已填充。

平台交互监测已两次出现gRPC等待超时及exec运行时替换，持久化过程及恢复检查在外部monitoring_disruptions.json。两次均核对原进程仍存活，没有把工具超时当成训练退出。

独立完成监督器PID 31764状态为WAITING_FOR_TRAINING。在本次训练实际退出后，将读取持有Windows进程句柄记录的真实退出码；仅在正常退出时执行postprocess.py：校验best和样本清单、重算同一完整验证集的六种方法、检查正式预测与加载一致性、生成曲线/报告并追加DEVELOPMENT_PROGRESS.md。此流程尚未实际完成，不能提前标为PASS。不会自动训练C或其他模型。

训练文件位于artifacts/stage4/full/O/retrieval_same_group_seed42/；完整命令、环境、标准输出/错误、监测、退出记录及完成监督位于artifacts/stage4/full_operations/O_retrieval_same_group_seed42/。未来正常交付报告路径为运行目录formal_training_report.md，目前不存在。

未完成：正式训练停止判定、最终best轮次及验证MSE、六方法完整指标表、最终训练曲线、正式预测检查和最终交付。多种子、C、完整方法对照和测试集评价也未执行；不标记整个第四阶段完成。


## 2026-11-17 4B-O意外中断核验与同目录续训（未完成）

22:24核对：原训练PID9124、启动器30092及完成监督31764均不存在；最后心跳为第1轮410/866批，无退出收据、FAILED.json或summary.json。中断原因及原退出码未知，不推断为正常结束或系统重启。

恢复核验PASS：42源码及884受保护文件SHA一致，依赖未变，完整O训练/验证清单相同；last分代9ea6e2c71d264153ab820c9812709c9c，epoch1/next_batch400/steps400，样本累计102400；Adam、随机状态、损失统计与游标可恢复，无best。训练缓存97948条完整行逐条校验通过，验证缓存0条；未提交缓存行按原实现重新计算。

22:34:59在原目录执行已授权resume（非new），完整配置保持O/full/42/retrieval/same_group、batch256/max100/patience10、每25批保存/每30秒进度。原目录和全部样本不变。真实训练PID6716记录resumed事件：epoch1/next_batch400/steps400，后续已推进至412批；监督PID16136持有进程句柄记录实际退出码，只在成功退出后执行完整验证与报告，不自动重试。原监督器已不存在，未重复启动后处理。

恢复命令：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli resume --run-dir artifacts/stage4/full/O/retrieval_same_group_seed42 --experiment O --kind full --seed 42 --variant retrieval --method same_group --checkpoint-every-batches 25 --progress-seconds 30 --confirm-full-training
```

命令、恢复核验、原last副本及新日志位于artifacts/stage4/full_operations/O_retrieval_same_group_seed42/recovery_20261117/。training_interruptions.json记录中断现场；原启动日志及监督状态副本保留。

仅补充外部监督/交付脚本的中断、分次耗时及计数口径，原脚本副本保留；src、算法、依赖、冻结产物和参数不变。累计训练计时不含中断前未提交工作及停机间隔，最终进程退出收据只代表恢复后的进程，不冒充全程耗时。

未完成训练及后处理，不能标为完成。未启动C、其他种子/模型或测试评价。一次PowerShell管道追加中文编码失败，仅修正本次新追加段，失败文本另存recovery_note_encoding_failed.txt；此前历史字节保持不变。


## 2026-11-18 4B-O首轮完整验证记录（训练仍在进行）

01:25，原运行目录的续训进程PID6716完成第1轮全部866批及完整验证，并进入第2轮。仍仅O/full/seed42/retrieval/same_group；未调整代码、依赖或训练参数，未启动C、其他种子/模型或测试集评价。

实际curve.json记录：归一化训练MSE=1.5887908120628869，原金额单位验证MSE=1271.4959705375052，best_epoch=1，bad_epochs=0，reference_masked=41758。本轮保存状态累计的数据准备时间16309.767790999787秒、优化步骤时间4.550252000393812秒、完整验证时间955.466477399983秒；这些累计量不含中断前未提交工作及停机间隔，不能当作完整墙钟时间。

首个best分代为0d9308d7480a45c584eb6107ad5ee28f，权重、元数据及训练/验证清单SHA均核验通过。模型版本55a132487e57f72a605139e9c3ad3554da8ddfed547510bdecc876ec3f9c26c6。独立对实际progress.jsonl中的验证前后缓存请求计数作差，得到98664次验证参考请求，与程序计算的完整验证清单98664条一致；完整清单含4111名客户。证据：recovery_20261117/first_epoch_observation.json。

00:58再次通过Windows原生进程句柄核对身份：原PID9124/30092/31764无法打开（WinError87），原退出码无法补取；当前训练PID6716、启动器15440及监督PID16136的创建时间与恢复记录一致且仍存活（259为STILL_ACTIVE，不是成功退出码）。证据：recovery_20261117/native_identity_check_20261118.json。

截至02:01的监测，第二轮到218/866批，最近提交断点为本轮209批/累计1075批；训练及唯一监督器存活，训练stderr为空，无FAILED.json或完整summary.json。首轮best独立保留，后续last继续按累计每25批保存。全部后续观察追加到recovery_20261117/monitor-observations.jsonl，训练心跳及每轮结果仍在运行目录progress.jsonl/curve.json。

当前仅完成1轮，不是正式训练完成报告。最终最佳轮次、六方法完整指标表、最终曲线及正式预测验收仍待训练按原早停/100轮条件结束后，由现有监督器执行。未重复启动后处理。


## 2026-11-18 11:45 对话监测恢复核对（训练仍在进行）

对话内wait返回aborted，随后旧exec cell 32已不存在。对话内最后一条观察为02:05:54，第2轮250批；11:45:58重新核对时，原续训PID6716、启动器15440、监督PID16136均仍存活，创建时间与22:34恢复记录一致。没有执行new、resume或重复启动后处理。

实际curve.json已记录完整完成8轮，正在第9轮（11:45快照为429/866批）。当时last提交为本轮422批/累计7350批，分代9b55cc9b805d476fbd67cbace7dc3b6c；最佳为第7轮，原金额单位验证MSE=1269.7982033260112，best分代3487e313b7c54126a5c57e11adcf6d89。第8轮未改善，尚未满足patience=10。

原监督状态WAITING_FOR_RESUMED_TRAINING；训练及监督stderr为空，无完整summary.json。外部内存记录持续更新，11:45实际进程观测峰值工作集1797808128字节。此处记录的是对话监测间隔，不是新的训练中断。

证据：monitoring_disruptions.json新增独立事件；recovery_20261117/dialogue_gap_recheck_20261118.json保存身份核对；训练过程仍由同一progress.jsonl、curve.json和分代提交清单记录。继续只监测本组O/full/seed42/retrieval/same_group，尚未形成完成报告。


# 4B-O：普通历史实验O、seed42正式训练报告

状态：实际完成；停止原因：early_stopping。完成17轮，best为第7轮，验证MSE=1269.798203326011。

本次仅O/full/42/retrieval/same_group；未运行C、其他种子、其他模型训练或测试集评价。算法、依赖、聚类及比例未改动。

## 命令、环境和完整样本

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli new --run-dir artifacts/stage4/full/O/retrieval_same_group_seed42 --experiment O --kind full --seed 42 --variant retrieval --method same_group --checkpoint-every-batches 25 --progress-seconds 30 --confirm-full-training
```

Python 3.12.14，PyTorch 2.15.0+cpu，CPU；PyTorch/BLAS各2线程；依赖与4A补强快照完全一致。batch256、max_epochs100、patience10；每25批保存、每30秒进度。

训练4111名客户，221680个普通合法切点对，1107718条可用历史模式记录；每轮221696次抽样、866批。验证4111名客户、98664条普通历史样本。训练原点59—112、验证原点119—142，预热拟合截止59，验证参考库冻结119。完整保存清单逐列与第二阶段源表相等。

## 验证结果

全部方法使用相同完整验证客户/切点/标签、原金额单位；先客户内平均，再客户等权。纯检索无参考时按既定规则回退Q均值。

| 方法 | MSE | RMSE | MAE | TotalAE |
| --- | ---: | ---: | ---: | ---: |
| model | 1269.798203326 | 35.634227974 | 19.546346655 | 66.283558951 |
| zero | 2082.737676617 | 45.637020900 | 27.743544396 | 194.204810772 |
| last | 2512.484412763 | 50.124688655 | 26.870928881 | 148.779967465 |
| q_mean | 1317.057694657 | 36.291289515 | 20.008974753 | 75.095572380 |
| q_median | 1345.777199345 | 36.684836095 | 19.032640868 | 75.227897916 |
| retrieval | 1618.657479772 | 40.232542547 | 22.737555425 | 80.697672121 |

本次模型验证MSE低于这五个无需训练基线；这仅是O单种子验证结果，不代表测试集效果。

## 实测训练曲线

![训练损失](training_loss.png)

![验证MSE](validation_mse.png)

| 轮次 | 归一化训练MSE | 原金额验证MSE | best轮次 |
| --- | ---: | ---: | ---: |
| 1 | 1.5887908121 | 1271.4959705375 | 1 |
| 2 | 1.5966800049 | 1270.5228819604 | 2 |
| 3 | 1.5679781314 | 1271.3554529560 | 2 |
| 4 | 1.5885816300 | 1270.3382922013 | 4 |
| 5 | 1.5660956004 | 1270.8009211932 | 4 |
| 6 | 1.5656944406 | 1270.5142996507 | 4 |
| 7 | 1.5726531410 | 1269.7982033260 | 7 |
| 8 | 1.5903401818 | 1271.9711296905 | 7 |
| 9 | 1.6246198597 | 1271.4458684972 | 7 |
| 10 | 1.5800807195 | 1269.9845754977 | 7 |
| 11 | 1.5631408931 | 1269.9575898345 | 7 |
| 12 | 1.5592078632 | 1271.0582125478 | 7 |
| 13 | 1.6010506605 | 1269.8131270434 | 7 |
| 14 | 1.5879860646 | 1270.9039990259 | 7 |
| 15 | 1.5853923974 | 1270.1886385328 | 7 |
| 16 | 1.5632855465 | 1270.1994515628 | 7 |
| 17 | 1.6115137641 | 1270.9647208776 | 7 |

完整未舍入曲线保存于curve.json/curve.csv。PNG由Windows内置.NET Charting离线绘制，不新增Python依赖。

## 验证集正式预测示例

以下三个示例均来自验证集t=119，按客户ID排序取前三名，非测试集；完整精度与追溯字段见validation_examples.json。

- C1000148617，t=119，客群：财富安享客户，稳健/进取：0.85/0.15。
  7日预测：[30.616753658056258, 29.61997206568718, 30.21632349014282, 29.336515536308287, 29.528329931497574, 29.92644619464874, 29.97181342959404]；7日实际正常消费：[8.45, 53.81, 12.67, 45.68, 13.54, 11.8, 40.29]。
- C100045114，t=119，客群：财富安享客户，稳健/进取：0.85/0.15。
  7日预测：[30.328966547250747, 30.05386116862297, 29.795231175422668, 30.361556169986724, 30.204146726131437, 29.726437252759933, 30.685189628601073]；7日实际正常消费：[43.64, 56.97, 35.67, 20.17, 27.1, 19.05, 4.74]。
- C1000699316，t=119，客群：财富增长客户，稳健/进取：0.65/0.35。
  7日预测：[6.153550864160061, 7.614426965415477, 7.458124935925007, 6.322957296967506, 7.6461664518713945, 8.54993882060051, 8.699907974004745]；7日实际正常消费：[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]。

共同模型版本：55a132487e57f72a605139e9c3ad3554da8ddfed547510bdecc876ec3f9c26c6。预测状态均OK、长度7、有限非负；加载前后逐元素一致，总额与逐日和在绝对1e-12/相对0容差下相等。业务名称仅按年龄顺序映射，不宣称完整财富阶段画像已验证。

## 时间、缓存和资源

最后一次实际Python训练进程创建至退出的墙钟时间：69592.691秒（Windows进程句柄GetProcessTimes）；保存断点累计时间加恢复后的训练循环及最终验证：76686.267秒。该累计值不含中断前未提交工作和停机间隔，不能当作所有进程总耗时。原进程实际退出时刻和退出码未捕获，总进程耗时未测得。独立完整验证复核：31.094秒。

实际Python训练进程PID=[6716, 9124]，观测到的进程生命周期峰值工作集=1816444928字节（1.692GiB），每30秒通过Windows PeakWorkingSet读取，共2550次；包含原生分配，但未采样间隔可能遗漏更高峰值。启动器PID的几MB内存不作为训练内存。

写报告前运行目录379138505字节，缓存369319419字节；最后一次训练进程缓存命中2762379、未命中904053，计数在恢复进程中从零开始，不能当作全运行累计。原进程计数见training_interruptions.json。完整行数/分区/磁盘量见formal_validation.json。

## 断点、文件及验收范围

本次发生1次已记录训练中断及恢复：原进程与原监督器均消失，退出码和原因未知；确认无活动写进程、源码/依赖/清单/last/缓存一致后，在原目录从第1轮第400批last续训，未使用best续训。原日志、断点副本和恢复命令保存在外部recovery_20261117/与training_interruptions.json。此前两次工具超时未中断当时训练，另见monitoring_disruptions.json。best分代：3487e313b7c54126a5c57e11adcf6d89；last分代：84e0b88c60a1440db6e373113e0ef9f0；提交序号623。通过提交清单定位并校验权重、元数据和样本清单，最终完整验证与示例均使用best。

运行目录：artifacts/stage4/full/O/retrieval_same_group_seed42/。checkpoint_manifest.json及checkpoint_previous.json为分代提交清单；checkpoints/保存分代权重及JSON；config.json保存正式配置/拟合/数据与代码版本；curve.json/.csv及两张PNG保存曲线；summary.json保存训练结果；formal_validation.json、validation_recheck.json和validation_examples.json保存独立复核。

外部启动/监测日志：artifacts/stage4/full_operations/O_retrieval_same_group_seed42/；原launch及日志完整保留，恢复后的命令、标准输出/错误、监督状态和进程退出记录位于recovery_20261117/。根runtime_exit.json/launcher_exit.json指向恢复后进程的真实退出结果，不代表原中断进程。runtime-memory.jsonl包含两次进程采样。运行内部progress.jsonl按PID保留全部心跳与checkpoint/轮末事件。仅外部监督和报告记录有所更新；训练代码、参数及依赖未变。

本次O单种子正式训练、完整验证、best保存加载和正式预测接口检查通过。尚未完成：C正式训练、42/43/44完整多种子实验、独立无检索和完整方法对照、测试集评价；M12/E08及跨实验完整验收仍未全部完成。没有把本次结果标为整个第四阶段全部完成。

本次交付后停止，不自动启动C实验。


## 4B-C正式训练启动记录（尚未完成）

2026-11-18（Asia/Shanghai）。用户明确授权唯一任务C/full/seed42/retrieval/same_group；batch256、max_epochs100、patience10、每25批保存、每30秒日志。本记录仅表示已启动，不表示训练或验收完成。

已读取算法规格、默认配置、开发历史、4A续训补强报告及4B-O正式报告。项目.venv仍为Python3.12.14、PyTorch2.15.0+cpu，CPU、PyTorch/BLAS各2线程；无依赖安装/升级，无src或默认配置修改，无聚类重拟合。O正式产物及监督记录已逐文件纳入本次保护哈希。

C实际ID划分2878/616/618，互不相交。具有合法训练窗口的客户2877名，普通切点对155136，历史模式记录775193（ordinary155136、Lc1为155136、Lc3为155100、Lc7为155017、Lc14为154804），每轮155136次抽样、606批。训练切点59—112。验证切点119—142，Lc1/3/7/14各616名客户、14784样本，共59136样本，四档均非空。C冻结供体2785名，金额尺度a=26.88；仅预热0—59拟合，验证候选截止119。验证/测试ID不参与拟合或供体库。

预检实际退出0，耗时30.976秒；磁盘可用291679649792字节，内存可用15620911104字节。证据为preflight.json、preflight-exit.json和environment.lock.txt。预检使用完整第二阶段合法表，并由LegalSamples重新断言ID、标签、历史和属性范围。

本次只新增C监督目录中的common.py、win_process.py、preflight.py、supervise.py、postprocess.py、plot_curves.ps1及辅助检查check_helpers.py。监督器持有真实训练进程句柄，采集内存和退出码，归档每轮已提交验证时点权重，训练后只启动一次后处理。各Lc曲线在训练后从归档权重复算，逐轮汇总必须与原训练曲线精确一致，不改变训练、采样或早停。

辅助归档检查未执行训练或优化步骤；初次因未创建归档父目录而失败，原合成夹具与helper_checks_initial_failure.json保留；只修正C辅助脚本，auxiliary_check_v2复核通过，包含归档只读、忽略轮中状态、按轮去重及拒绝O元数据。真实C初始化归档为零优化步，数值哈希应与独立torch.manual_seed(42)初始化核对。

实际由PowerShell启动隐藏监督器，监督器在项目根目录执行以下训练命令（完整args、PID、时间见attempts/new_001/launch.json）：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m banksim.forecast_train_cli new --run-dir artifacts/stage4/full/C/retrieval_same_group_seed42 --experiment C --kind full --seed 42 --variant retrieval --method same_group --checkpoint-every-batches 25 --progress-seconds 30 --confirm-full-training
```

首次启动2026-11-18 20:30:05；训练启动器PID29652，实际训练PID37072，实际监督器PID27180（监督启动器4480）。20:31核验训练已进入第1轮，零步断点904446734d1548fabc5aff0b9f1c78d3已保存并归档，训练及监督错误日志为空。后续以实际创建时间和新日志确认身份，不因聊天断流重复启动。正式报告尚未生成，最终完成状态须同时检查训练退出和后处理结果。

运行目录artifacts/stage4/full/C/retrieval_same_group_seed42/；监督目录artifacts/stage4/full_operations/C_retrieval_same_group_seed42/。本次不启动其他种子、其他模型、测试集评价或界面开发。
