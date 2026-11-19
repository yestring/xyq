# 数据来源与使用说明

作者发布源：https://www.kaggle.com/datasets/ealaxi/banksim1

作者：Edgar Lopez-Rojas。公开数据集版本1，下载核验日期2026年11月15日。发布元数据标注CC BY-NC-SA 4.0，许可说明：https://creativecommons.org/licenses/by-nc-sa/4.0/ 。保留来源及作者署名；后续使用和分发应遵循该许可。

原论文：Lopez-Rojas, E. A. & Axelsson, S. (2014). BankSim: A Bank Payments Simulator for Fraud Detection Research. EMSS 2014, pp.144–152。https://www.msc-les.org/proceedings/emss/2014/EMSS2014_144.pdf

data/raw/bs140513_032310.csv是该发布版本的完整交易表，原样复制，未筛选、未抽样、未更改引号。文件48,986,035字节，SHA-256为e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377。

原发布包还含同批交易的网络边表bsNET140513_032310.csv。本算法不使用它，因此交接包省略该冗余表示，不能将其追加为新交易。data/reference/source_metadata.json保存原发布条目元数据；normal_data_audit.json仅为本地核验结果。

本包没有清洗后的训练数据。剔除欺诈和日聚合留给编码第一阶段实现，并对照审计总数验证。所有派生输出必须另外保存，不覆盖raw。
