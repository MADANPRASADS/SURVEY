#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


md = pd.read_csv("data-final.csv", sep='\t')


# In[4]:


data = md.copy()


# In[5]:


data.drop(data.columns[50:107], axis=1, inplace=True)
data.drop(data.columns[51:], axis=1, inplace=True)


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data.describe()


# In[11]:


print(data.isnull().values.sum())


# In[13]:


ext_questions = {'EXT1' : 'I am the life of the party',
                 'EXT2' : 'I dont talk a lot',
                 'EXT3' : 'I feel comfortable around people',
                 'EXT4' : 'I keep in the background',
                 'EXT5' : 'I start conversations',
                 'EXT6' : 'I have little to say',
                 'EXT7' : 'I talk to a lot of different people at parties',
                 'EXT8' : 'I dont like to draw attention to myself',
                 'EXT9' : 'I dont mind being the center of attention',
                 'EXT10': 'I am quiet around strangers'}

est_questions = {'EST1' : 'I get stressed out easily',
                 'EST2' : 'I am relaxed most of the time',
                 'EST3' : 'I worry about things',
                 'EST4' : 'I seldom feel blue',
                 'EST5' : 'I am easily disturbed',
                 'EST6' : 'I get upset easily',
                 'EST7' : 'I change my mood a lot',
                 'EST8' : 'I have frequent mood swings',
                 'EST9' : 'I get irritated easily',
                 'EST10': 'I often feel blue'}

agr_questions = {'AGR1' : 'I feel little concern for others',
                 'AGR2' : 'I am interested in people',
                 'AGR3' : 'I insult people',
                 'AGR4' : 'I sympathize with others feelings',
                 'AGR5' : 'I am not interested in other peoples problems',
                 'AGR6' : 'I have a soft heart',
                 'AGR7' : 'I am not really interested in others',
                 'AGR8' : 'I take time out for others',
                 'AGR9' : 'I feel others emotions',
                 'AGR10': 'I make people feel at ease'}

csn_questions = {'CSN1' : 'I am always prepared',
                 'CSN2' : 'I leave my belongings around',
                 'CSN3' : 'I pay attention to details',
                 'CSN4' : 'I make a mess of things',
                 'CSN5' : 'I get chores done right away',
                 'CSN6' : 'I often forget to put things back in their proper place',
                 'CSN7' : 'I like order',
                 'CSN8' : 'I shirk my duties',
                 'CSN9' : 'I follow a schedule',
                 'CSN10' : 'I am exacting in my work'}

opn_questions = {'OPN1' : 'I have a rich vocabulary',
                 'OPN2' : 'I have difficulty understanding abstract ideas',
                 'OPN3' : 'I have a vivid imagination',
                 'OPN4' : 'I am not interested in abstract ideas',
                 'OPN5' : 'I have excellent ideas',
                 'OPN6' : 'I do not have a good imagination',
                 'OPN7' : 'I am quick to understand things',
                 'OPN8' : 'I use difficult words',
                 'OPN9' : 'I spend time reflecting on things',
                 'OPN10': 'I am full of ideas'}


EXT = [column for column in data if column.startswith('EXT')]
EST = [column for column in data if column.startswith('EST')]
AGR = [column for column in data if column.startswith('AGR')]
CSN = [column for column in data if column.startswith('CSN')]
OPN = [column for column in data if column.startswith('OPN')]


# In[14]:


def questionAnswers(groupName, questions, color):
    plt.figure(figsize=(40,60))
    for i in range(1, 11):
        plt.subplot(10,5,i)             
        plt.hist(data[groupName[i-1]], bins=10, color= color) 
        plt.title(questions[groupName[i-1]], fontsize=18)


# In[38]:


print('Extroversion Personality')
questionAnswers(EXT, ext_questions, 'red')


# In[39]:


print('Neuroticism Personality')
questionAnswers(EST, est_questions, 'grey')


# In[17]:


print('Agreeable Personality')
questionAnswers(AGR, agr_questions, 'orange')


# In[18]:


print('Conscientious Personality')
questionAnswers(CSN, csn_questions, 'blue')


# In[19]:


print('Openness Personality')
questionAnswers(OPN, opn_questions, 'green')


# #**Clustering:**

# # New Section

# In[20]:


from sklearn.cluster import KMeans

new_data = data.drop('country', axis=1);

kmeans = KMeans(n_clusters=5);
new_data.dropna(inplace=True)
k_fit = kmeans.fit(new_data);               


# In[21]:


pd.options.display.max_columns = 50;
predictions = k_fit.labels_
new_data['Clusters'] = predictions;
new_data.head()


# In[22]:


x = new_data.iloc[:, 0:50] 
print(x) 
y = new_data.iloc[:, 50:51]  
print(y)


# In[23]:


from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train.values.ravel())


y_pred = clf.predict(X_test)


# In[25]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[26]:


print(y_pred)


# In[27]:


new_data.Clusters.value_counts()


# In[28]:


pd.options.display.max_columns=50
new_data.groupby('Clusters').mean()


# **Checking the pattern after grouping**

# In[29]:


list_of_col = list(new_data)
ext = list_of_col[0:10]
est = list_of_col[10:20]
agr = list_of_col[20:30]
csn = list_of_col[30:40]
opn = list_of_col[40:50]

sum_data = pd.DataFrame()
sum_data['extroversion'] = new_data[ext].sum(axis=1)/10
sum_data['neurotic'] = new_data[est].sum(axis=1)/10
sum_data['agreeable'] = new_data[agr].sum(axis=1)/10
sum_data['conscientious'] = new_data[csn].sum(axis=1)/10
sum_data['open'] = new_data[opn].sum(axis=1)/10
sum_data['clusters'] = predictions
sum_data.groupby('clusters').mean()


# In[30]:


dataclusters = sum_data.groupby('clusters').mean()
plt.figure(figsize=(22,6))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(dataclusters.columns, dataclusters.iloc[:, i], color='green', alpha=0.2)
    plt.plot(dataclusters.columns, dataclusters.iloc[:, i], color='red')
    plt.title('Cluster ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4);


# In[31]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(new_data)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = predictions
df_pca.head()


# In[32]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
plt.title('Personality Clusters after PCA');


# In[42]:


items_and_questions = {
    "EXT1": "I am the life of the party.",
    "EXT2": "I don't talk a lot.",
    "EXT3": "I feel comfortable around people.",
    "EXT4": "I keep in the background.",
    "EXT5": "I start conversations.",
    "EXT6": "I have little to say.",
    "EXT7": "I talk to a lot of different people at parties.",
    "EXT8": "I don't like to draw attention to myself.",
    "EXT9": "I don't mind being the center of attention.",
    "EXT10": "I am quiet around strangers.",
    "EST1": "I get stressed out easily.",
    "EST2": "I am relaxed most of the time.",
    "EST3": "I worry about things.",
    "EST4": "I seldom feel blue.",
    "EST5": "I am easily disturbed.",
    "EST6": "I get upset easily.",
    "EST7": "I change my mood a lot.",
    "EST8": "I have frequent mood swings.",
    "EST9": "I get irritated easily.",
    "EST10": "I often feel blue.",
    "AGR1": "I feel little concern for others.",
    "AGR2": "I am interested in people.",
    "AGR3": "I insult people.",
    "AGR4": "I sympathize with others' feelings.",
    "AGR5": "I am not interested in other people's problems.",
    "AGR6": "I have a soft heart.",
    "AGR7": "I am not really interested in others.",
    "AGR8": "I take time out for others.",
    "AGR9": "I feel others' emotions.",
    "AGR10": "I make people feel at ease.",
    "CSN1": "I am always prepared.",
    "CSN2": "I leave my belongings around.",
    "CSN3": "I pay attention to details.",
    "CSN4": "I make a mess of things.",
    "CSN5": "I get chores done right away.",
    "CSN6": "I often forget to put things back in their proper place.",
    "CSN7": "I like order.",
    "CSN8": "I shirk my duties.",
    "CSN9": "I follow a schedule.",
    "CSN10": "I am exacting in my work.",
    "OPN1": "I have a rich vocabulary.",
    "OPN2": "I have difficulty understanding abstract ideas.",
    "OPN3": "I have a vivid imagination.",
    "OPN4": "I am not interested in abstract ideas.",
    "OPN5": "I have excellent ideas.",
    "OPN6": "I do not have a good imagination.",
    "OPN7": "I am quick to understand things.",
    "OPN8": "I use difficult words.",
    "OPN9": "I spend time reflecting on things.",
    "OPN10": "I am full of ideas."
}


responses = {}


for item, question in items_and_questions.items():
    while True:
        try:
            response = int(input(f"{question}\nRate on a scale from 1 to 5: "))
            if response < 1 or response > 5:
                raise ValueError("Rating should be between 1 and 5.")
            responses[item] = response
            break
        except ValueError as e:
            print(e)


testing_data = pd.DataFrame([responses])

print(testing_data)


# In[34]:


print("supervised prediction is" ,clf.predict(testing_data))


# 

# In[35]:


my_personality = k_fit.predict(testing_data)
print('My personality cluster: ', my_personality)


# **Sum of all row values.**

# In[36]:


my_list = list(testing_data)
ext = my_list[1:10]
est = my_list[10:20]
agr = my_list[20:30]
csn = my_list[30:40]
opn = my_list[40:50]

sums = pd.DataFrame()
sums['extroversion'] = testing_data[ext].sum(axis=1)/10
sums['neurotic'] = testing_data[est].sum(axis=1)/10
sums['agreeable'] = testing_data[agr].sum(axis=1)/10
sums['conscientious'] = testing_data[csn].sum(axis=1)/10
sums['openness'] = testing_data[opn].sum(axis=1)/10
sums['Cluster'] = my_personality
print('Sum of Personality columns:')
sums


# In[37]:


sum_new = sums.drop('Cluster', axis=1)
plt.bar(sum_new.columns, sum_new.iloc[0,:], color='blue', alpha=0.5)
plt.plot(sum_new.columns, sum_new.iloc[0,:], color='red')
plt.title('Cluster')
plt.xticks(rotation=45)
plt.ylim(0,4);


# In[ ]:




