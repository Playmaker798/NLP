# NLP
自然语言处理课程项目
🛠️ 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+ 
- 将数据文件放置在 `mydata/` 目录下：
- `cnews.train.txt` - 训练集
- `cnews.test.txt` - 测试集  
- `cnews.val.txt` - 验证集
- 数据格式：每行一条数据，格式为 `标签\t文本内容`
- 准备模型
将 BERT 模型文件放置在 `models/bert-base-chinese/` 目录下，或程序会自动从网络下载。
