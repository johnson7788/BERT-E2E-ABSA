import pymongo
import json
import random
import os

def db2local(save_file):
  """
  从MongoDB 获取数据，保存到save_file中
  :param save_file:
  :return:
  """
  # 配置client
  client = pymongo.MongoClient("192.168.50.139", 27017)
  # 设置database
  db = client['ai-corpus']
  # 选择哪个collections
  collection = db['as_corpus']
  mydoc = collection.find({})
  with open(save_file, 'w') as f:
    for x in mydoc:
      x.pop('_id')
      content = json.dumps(x)
      f.write(content+ '\n')
  print(f"文件已生成{save_file}")

def split_all(save_file, train_rate=0.8, dev_rate=0.1, test_rate=0.1):
  """
  拆分成90%训练集，5%开发集和5%测试集
  :param save_file:
  :param train_rate: float
  :param dev_rate:
  :param test_rate:
  :return:
  """
  random.seed(30)
  with open(save_file, 'r') as f:
    lines = f.readlines()
  random.shuffle(lines)
  total = len(lines)
  train_num = int(total*train_rate)
  dev_num = int(total*dev_rate)
  test_num = int(total*test_rate)
  train_file = os.path.join(os.path.dirname(save_file),'train.txt')
  dev_file = os.path.join(os.path.dirname(save_file),'dev.txt')
  test_file = os.path.join(os.path.dirname(save_file),'test.txt')
  with open(train_file, 'w') as f:
    for x in lines[:train_num]:
      f.write(x)
  with open(dev_file, 'w') as f:
    for x in lines[train_num:train_num+dev_num]:
      f.write(x)
  with open(test_file, 'w') as f:
    for x in lines[train_num+dev_num:]:
      f.write(x)
  print(f"文件已生成\n {train_file}, 样本数: {train_num} \n {dev_file}, 样本数: {dev_num} \n {test_file}, 样本数: {test_num}")

def pre_process(save_file, new_file):
  """
  处理成和rest15一样的文件, 就是对所有的字进行BIOES分类, 标签类似如下， 是联合任务 {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                        'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                        'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
  :param save_file:
  :param new_file: 存储到新文件
  :return: 存储到文件，格式是： 句子####字1=O 字2=B-POS ...
  """
  #原始文件中的sScore的映射方式
  class2id = {
    "NEG": 0,
    "NEU": 1,
    "POS":2,
  }
  id2class = {value:key for key,value in class2id.items()}
  with open(save_file, 'r') as f:
    lines = f.readlines()
  #打印多少条样本
  print_example = 10
  #总数据量
  total = 0
  with open(new_file, 'w') as f:
    for line in lines:
      line_chinese = json.loads(line)
      #分隔符是####， 防止歧义，如果存在####的话，事先去掉原有内容的####分隔符
      content = line_chinese["content"].replace("####","")
      des_line_prefix = content + "####"
      total_words_num = len(line_chinese["content"])
      tags = ["O"] * total_words_num
      #如果这个句子没有aspect，那就过滤掉
      if not line_chinese["aspect"]:
        continue
      for aspect in line_chinese["aspect"]:
        start = aspect["start"]
        end = aspect["end"]
        sScore = aspect["sScore"]
        #单个单词
        sentiment = id2class[sScore]
        if end - start == 1:
          tags[start] = "S-" + sentiment
        elif end - start == 2:
          tags[start] = "B-" + sentiment
          tags[start+1] = "E-" + sentiment
        elif end - start >2:
          tags[start] = "B-" + sentiment
          tags[start+1:end-1] = ["I-" + sentiment] * (end-start-2)
          tags[end - 1] = "E-" + sentiment
        else:
          print(f"{line}line_chinese")
      des_line_suffix = ""
      for word, tag in zip(line_chinese["content"], tags):
        des_line_suffix += f"{word}={tag} "
      des_line = des_line_prefix + des_line_suffix.strip()
      total += 1
      if print_example > 0:
        print(des_line)
        print_example -= 1
      f.write(des_line + "\n")
  print(f"文件已生成{new_file}, 总数据量是{total}")

def only_sentiment_process(save_file, new_file):
  """
  处理成和rest15一样的文件,单纯的情感分类，标签是 整个单词=(起始位置，结束位置，情感), {'NEG':0, 'NEU':1, 'POS':2}， 每个句子只有一个单词的预测
  :param save_file:
  :param new_file: 存储到新文件
  :return: 存储到文件，格式是： 句子####单词=(3,5,'NEG')
                            句子####单词=(10,15,'NEU')
  """
  #原始文件中的sScore的映射方式
  class2id = {
    "NEG": 0,
    "NEU": 1,
    "POS":2,
  }
  id2class = {value:key for key,value in class2id.items()}
  with open(save_file, 'r') as f:
    lines = f.readlines()
  #打印多少条样本
  print_example = 10
  #总数据量
  total = 0
  with open(new_file, 'w') as f:
    for line in lines:
      line_chinese = json.loads(line)
      #分隔符是####， 防止歧义，如果存在####的话，事先去掉原有内容的####分隔符
      content = line_chinese["content"].replace("####","")
      # 对content进行split处理，把它分成每个字
      split_content = [c for c in content]
      newcontent = " ".join(split_content)
      des_line_prefix = newcontent + "####"
      #如果这个句子没有aspect，那就过滤掉
      if not line_chinese["aspect"]:
        continue
      for aspect in line_chinese["aspect"]:
        aspectTerm = aspect["aspectTerm"]
        sScore = aspect["sScore"]
        start = aspect["start"]
        end = aspect["end"]
        #单个单词
        sentiment = id2class[sScore]
        #验证一下单词的位置是否在newcontent中位置对应
        aspectTerm_insentence = "".join(split_content[start:end])
        if not aspectTerm == aspectTerm_insentence:
          raise Exception(f"单词在句子中位置对应不上，请检查,句子行数{total}, 句子是{line_chinese}")
        des_line_suffix = f"{aspectTerm}=({start},{end},{sentiment})"
        des_line = des_line_prefix + des_line_suffix.strip()
        if print_example > 0:
          print(des_line)
          print_example -= 1
        total += 1
        f.write(des_line + "\n")
  print(f"文件已生成{new_file}, 总数据量是{total}")

def check_data(save_file):
  """
  没啥用，检查下数据
  :param save_file:
  :return:
  """
  with open(save_file, 'r') as f:
    lines = f.readlines()

  without_aspect = []
  for line in lines:
    line_chinese = json.loads(line)
    if not line_chinese["aspect"]:
      without_aspect.append(line_chinese)
      print(line_chinese)
  print(f"没有aspect的数量是{len(without_aspect)}")

def clean_cache():
  """
  删除../data/cosmetics/cached* 文件
  :return:
  """
  os.system("rm -rf ../data/cosmetics/cached*")
  os.system("rm -rf ../log/*")
  os.system("rm -rf ../run/*")

if __name__ == '__main__':
  save_file = "../data/cosmetics/all.txt"
  new_file = "../data/cosmetics/final_all.txt"
  # db2local(save_file)
  pre_process(save_file,new_file)
  # only_sentiment_process(save_file,new_file)
  split_all(new_file,train_rate=0.8,dev_rate=0.1,test_rate=0.1)
  # split_all(new_file,train_rate=0.9,dev_rate=0.05,test_rate=0.05)
  # check_data(save_file)
  clean_cache()