import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers


class MovieLen(object):
    def __init__(self):
        # 声明每个数据文件的路径
        usr_info_path = "users.dat"
        rating_path = "ratings.dat"
        movie_info_path = "movies.dat"

        self.usr_info, self.max_usr_id, self.max_usr_age, self.max_usr_job = self.get_usr_info(usr_info_path)

        self.movie_info, self.movie_cat, self.movie_titles, self.max_mov_id = self.get_movie_info(movie_info_path)

        self.rating_info = self.get_rating_info(rating_path)

        # 构建数据集
        self.dataset = self.get_dataset(usr_info=self.usr_info, rating_info=self.rating_info,
                                        movie_info=self.movie_info)

        # 划分数据及，获得数据加载器
        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.9)]
        self.valid_dataset = self.dataset[int(len(self.dataset) * 0.9):]

        print("##Total dataset instances: ", len(self.dataset))
        print("##MovieLens dataset information: \n usr num: {}\n"
              "movies num: {}".format(len(self.usr_info), len(self.movie_info)))

    def get_usr_info(self, path):
        def gender2num(gender):
            return 1 if gender == 'F' else 0

        data = open(path).readlines()
        usr_info = {}
        max_usr_id = 0
        max_usr_age = 0
        max_usr_job = 0
        # 按行索引数据
        for item in data:
            item = item.strip().split("::")
            usr_id = item[0]
            # 将字符数据转成数字并保存在字典中
            usr_info[usr_id] = {'usr_id': int(usr_id),
                                'gender': gender2num(item[1]),
                                'age': int(item[2]),
                                'job': int(item[3])}
            max_usr_id = max(max_usr_id, int(usr_id))
            max_usr_age = max(max_usr_age, int(item[2]))
            max_usr_job = max(max_usr_job, int(item[3]))
        return usr_info, max_usr_id, max_usr_age, max_usr_job

    def get_movie_info(self, path):
        # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
        data = open(path, 'r', encoding="ISO-8859-1").readlines()
        # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
        movie_info, movie_titles, movie_cat = {}, {}, {}
        # 对电影名字、类别中不同的单词计数
        t_count, c_count = 1, 1

        count_tit = {}
        # 按行读取数据并处理
        for item in data:
            item = item.strip().split("::")
            v_id = item[0]
            v_title = item[1][:-7]
            cats = item[2].split('|')
            v_year = item[1][-5:-1]

            titles = v_title.split()
            # 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
            for t in titles:
                if t not in movie_titles:
                    movie_titles[t] = t_count
                    t_count += 1
            # 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
            for cat in cats:
                if cat not in movie_cat:
                    movie_cat[cat] = c_count
                    c_count += 1
            # 补0使电影名称对应的列表长度为15
            v_tit = [movie_titles[k] for k in titles]
            while len(v_tit) < 15:
                v_tit.append(0)
            # 补0使电影种类对应的列表长度为6
            v_cat = [movie_cat[k] for k in cats]
            while len(v_cat) < 6:
                v_cat.append(0)
            # 保存电影数据到movie_info中
            movie_info[v_id] = {'mov_id': int(v_id),
                                'title': v_tit,
                                'category': v_cat,
                                'years': int(v_year)}
            max_mov_id = max(list(map(int, movie_info.keys())))
        return movie_info, movie_cat, movie_titles, max_mov_id

    def get_rating_info(self, path):
        # 读取文件里的数据
        data = open(path).readlines()
        # 将数据保存在字典中并返回
        rating_info = {}
        for item in data:
            item = item.strip().split("::")
            usr_id, movie_id, score = item[0], item[1], item[2]
            if usr_id not in rating_info.keys():
                rating_info[usr_id] = {movie_id: float(score)}
            else:
                rating_info[usr_id][movie_id] = float(score)
        return rating_info

    def get_dataset(self, usr_info, rating_info, movie_info):
        trainset = []
        for usr_id in rating_info.keys():
            usr_ratings = rating_info[usr_id]
            for movie_id in usr_ratings:
                trainset.append({'usr_info': usr_info[usr_id],
                                 'mov_info': movie_info[movie_id],
                                 'scores': usr_ratings[movie_id]})
        return trainset

    def load_data(self, dataset, mode='train'):
        BATCHSIZE = 256
        data_length = len(dataset)
        index_list = list(range(data_length))

        # 定义数据迭代加载器
        def data_generator():
            # 训练模式下，打乱训练数据
            if mode == 'train':
                random.shuffle(index_list)
            # 声明每个特征的列表
            usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
            mov_id_list, mov_tit_list, mov_cat_list = [], [], []
            score_list = []
            # 索引遍历输入数据集
            for idx, i in enumerate(index_list):
                # 获得特征数据保存到对应特征列表中
                usr_id_list.append(dataset[i]['usr_info']['usr_id'])
                usr_gender_list.append(dataset[i]['usr_info']['gender'])
                usr_age_list.append(dataset[i]['usr_info']['age'])
                usr_job_list.append(dataset[i]['usr_info']['job'])

                mov_id_list.append(dataset[i]['mov_info']['mov_id'])
                mov_tit_list.append(dataset[i]['mov_info']['title'])
                mov_cat_list.append(dataset[i]['mov_info']['category'])

                score_list.append(int(dataset[i]['scores']))

                # 如果读取的数据量达到当前的batch大小，就返回当前批次
                if len(usr_id_list) == BATCHSIZE:
                    # 转换列表数据为数组形式，reshape到固定形状
                    usr_id_arr = np.expand_dims(np.array(usr_id_list), axis=-1)
                    usr_gender_arr = np.expand_dims(np.array(usr_gender_list), axis=-1)
                    usr_age_arr = np.expand_dims(np.array(usr_age_list), axis=-1)
                    usr_job_arr = np.expand_dims(np.array(usr_job_list), axis=-1)

                    mov_id_arr = np.expand_dims(np.array(mov_id_list), axis=-1)
                    mov_cat_arr = np.reshape(np.array(mov_cat_list), [BATCHSIZE, 6, 1]).astype(np.int64)
                    mov_tit_arr = np.reshape(np.array(mov_tit_list), [BATCHSIZE, 15, 1]).astype(np.int64)

                    scores_arr = np.reshape(np.array(score_list), [-1, 1]).astype(np.float32)

                    # 放回当前批次数据
                    yield [usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr], [mov_id_arr, mov_cat_arr,
                                                                                   mov_tit_arr], scores_arr

                    # 清空数据
                    usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
                    mov_id_list, mov_tit_list, mov_cat_list, score_list = [], [], [], []

        return data_generator
# Dataset=MovieLen()

class Pro(tf.keras.layers.Layer):
    def __init__(self):
        super(Pro, self).__init__()
        Dataset = MovieLen()
        #         self.Dataset = Dataset
        #         self.trainset = self.Dataset.train_dataset
        #         self.valset = self.Dataset.valid_dataset
        #         self.train_loader = self.Dataset.load_data(dataset=self.trainset, mode='train')
        #         self.valid_loader = self.Dataset.load_data(dataset=self.valset, mode='valid')

        #     def embedding_usr(self,usr_var):
        #         usr_id, usr_gender, usr_age, usr_job = usr_var

        # 对用户ID做映射，并紧接着一个FC层
        self.usr_emb = layers.Embedding(Dataset.max_usr_id + 1, 32)
        self.usr_fc = layers.Dense(32, activation='relu')
        # 对用户性别信息做映射，并紧接着一个FC层
        self.usr_gender_emb = layers.Embedding(2, 16)
        self.usr_gender_fc = layers.Dense(16, activation='relu')
        # 对用户年龄信息做映射，并紧接着一个FC层
        self.usr_age_emb = layers.Embedding(Dataset.max_usr_age + 1, 16)
        self.usr_age_fc = layers.Dense(16, activation='relu')
        # 对用户职业信息做映射，并紧接着一个FC层
        self.usr_job_emb = layers.Embedding(Dataset.max_usr_job + 1, 16)
        self.usr_job_fc = layers.Dense(16, activation='relu')
        # 新建一个FC层，用于整合用户数据信息
        self.usr_combined = layers.Dense(200, activation='tanh')

        self.mov_emb = layers.Embedding(Dataset.max_mov_id + 1, 32)
        self.mov_fc = layers.Dense(32, activation='relu')
        # 对电影类别做映射
        self.mov_cat_emb = layers.Embedding(len(Dataset.movie_cat) + 1, 32)
        self.mov_cat_fc = layers.Dense(32, activation='relu')
        # 对电影名称做映射
        self.mov_title_emb = layers.Embedding(len(Dataset.movie_titles) + 1, 32)
        self.mov_title_conv = layers.Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same',
                                            activation='relu')
        self.mov_title_conv2 = layers.Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same',
                                             activation='relu')
        self.mov_concat_embed = layers.Dense(200, activation='tanh')

    # 将用户的ID数据经过embedding和FC计算，得到的特征保存在feats_collect中
    def call(self, usr_var, mov_var):
        usr_id, usr_gender, usr_age, usr_job = usr_var
        mov_id, mov_cat, mov_title = mov_var

        feats_collect = []
        usr_id = self.usr_emb(usr_id)
        usr_id = self.usr_fc(usr_id)
        feats_collect.append(usr_id)

        # 计算用户的性别特征，并保存在feats_collect中
        usr_gender = self.usr_gender_emb(usr_gender)
        usr_gender = self.usr_gender_fc(usr_gender)
        feats_collect.append(usr_gender)

        usr_age = self.usr_age_emb(usr_age)
        usr_age = self.usr_age_fc(usr_age)
        feats_collect.append(usr_age)

        # 计算用户的职业特征，并保存在feats_collect中
        usr_job = self.usr_job_emb(usr_job)
        usr_job = self.usr_job_fc(usr_job)
        feats_collect.append(usr_job)

        # 将用户的特征级联，并通过FC层得到最终的用户特征
        usr_feat = layers.concatenate(feats_collect, axis=-1)
        usr_feat = self.usr_combined(usr_feat)

        feats_collect1 = []
        mov_id = self.mov_emb(mov_id)
        mov_id = self.mov_fc(mov_id)
        feats_collect1.append(mov_id)

        mov_cat = self.mov_cat_emb(mov_cat)
        mov_cat = tf.reduce_sum(mov_cat, axis=1)
        mov_cat = self.mov_cat_fc(mov_cat)
        feats_collect1.append(mov_cat)

        mov_title = self.mov_title_emb(mov_title)
        mov_title = self.mov_title_conv2(self.mov_title_conv(mov_title))
        mov_title = tf.reduce_sum(mov_title, axis=2)
        #         mov_title = layers.ReLU(mov_title.any())
        mov_title = tf.reshape(mov_title, [256, 1, -1])
        feats_collect1.append(mov_title)

        # 使用一个全连接层，整合所有电影特征，映射为一个200维的特征向量
        mov_feat = layers.concatenate(feats_collect1, axis=-1)
        mov_feat = self.mov_concat_embed(mov_feat)

        return usr_feat, mov_feat

class Mo(tf.keras.Model):
    def __init__(self):
        super(Mo,self).__init__()
        self.pro=Pro()
    def call(self, usr_var, mov_var):
        # 计算用户特征和电影特征
        usr_feat,mov_feat= self.pro(usr_var,mov_var)
#         mov_feat = self.embedding_movie(mov_var)
        # 根据计算的特征计算相似度
        usr_norm=tf.sqrt(tf.reduce_sum(tf.square(usr_feat),axis=-1))
        mov_norm=tf.sqrt(tf.reduce_sum(tf.square(mov_feat),axis=-1))
        res=tf.reduce_sum(tf.multiply(usr_feat, mov_feat),axis=-1)/(usr_norm*mov_norm)
#         res = tf.keras.metrics.CosineSimilarity(usr_feat.all(), mov_feat.all())
        # 将相似度扩大范围到和电影评分相同数据范围
        res = res*5
        return usr_feat, mov_feat, res


def train(model):
    # 配置训练参数
    use_gpu = False
    lr = 0.01
    Epoches = 10
    #     model.train()
    # 获得数据读取器
    #     data_loader = model.train_loader
    # 使用adam优化器，学习率使用0.01
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(0, Epoches):
        Dataset = MovieLen()
        trainset = Dataset.train_dataset
        valset = Dataset.valid_dataset
        train_loader = Dataset.load_data(dataset=trainset, mode='train')
        valid_loader = Dataset.load_data(dataset=valset, mode='valid')
        for idx, data in enumerate(train_loader()):
            with tf.GradientTape() as tape:
                # 获得数据，并转为动态图格式
                usr, mov, score = data
                #                 usr_v = [var for var in usr]
                #                 mov_v = [var  for var in mov]
                #                 scores_label = score
                # 计算出算法的前向计算结果
                usr_feat, mov_feat, scores_predict = model(usr, mov)

                # 计算loss
                loss = tf.keras.losses.MSE(scores_predict, score)
                loss = tf.math.reduce_mean(loss)
                if idx % 500 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, loss.numpy()))
            gradients = tape.gradient(loss, model.trainable_variables)  # 计算梯度
            opt.apply_gradients(zip(gradients, model.trainable_variables))  # 梯度更新
        return loss


#                     model.clear_gradients()
# 每个epoch 保存一次模型
#             tf.saved_model.save(model.state_dict(), './checkpoint/epoch' + str(epoch))


# use_poster, use_mov_title, use_mov_cat, use_age_job = False, True, True, True
# model = Model('Recommend', use_poster, use_mov_title, use_mov_cat, use_age_job)
model = Mo()
loss = train(model)