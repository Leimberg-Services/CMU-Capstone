#!/usr/bin/env python
# coding: utf-8

# # Upgrade and Install Packages If Needed

# In[ ]:


#!pip install wordcloud
#!pip install threadpoolctl --upgrade


# # Load Packages

# In[1]:


import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from wordcloud import WordCloud
from wordcloud import STOPWORDS


# # Import Training Data and Drop Unessesary Columns

# In[ ]:


#import training data - csv
#paste in file location starting with 'C:\\ and ending in .csv
data = pd.read_csv('C:\\ .csv')

#import training data - excel
#paste in file location starting with r'C:\\ and ending in .csv'
#data = pd.read_excel(r'C: .xlsx')


# In[54]:


#drop a range of columns
data = data.drop(data.loc[:, 'Law':'Technology/IT'].columns, axis=1)
#drop rows based on having a specifis value in a column
data.drop(data.index[data['Company Field'] == 'Blank/Individual'], inplace = True)
data.drop(data.index[data['chatgpt description']=='no description, blank company or individual'], inplace = True)
data.drop(data.index[data['chatgpt description']== 'no description, blank company or individual CPA'], inplace = True)
data.drop(data.index[data['chatgpt description']== 'no description, blank company or law individual'], inplace = True)
#display information on variables now in dataset
data.info()


# In[55]:


#Larger Datasets can take a while to load. Save the imported data set to another name to run instead of importing again if a mistake is made later on. 
lister_fulltrain = data
#view the first 5 rows
lister_fulltrain.head()


# # Preprocess Training Text Data and Create Document Inputs for Model 

# In[56]:


# Remove punctuation
lister_fulltrain['chatgpt description2'] = ""
lister_fulltrain['chatgpt description2'] = lister_fulltrain['chatgpt description'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
lister_fulltrain['chatgpt description processed'] = \
lister_fulltrain['chatgpt description processed'] = lister_fulltrain['chatgpt description2'].map(lambda x: x.lower())
# Print out the first rows to see if looks right
lister_fulltrain['chatgpt description processed'].head()


# In[57]:


#process short law description text
# Remove punctuation
lister_fulltrain['law short2'] = ""
lister_fulltrain['law short2'] = lister_fulltrain['law short'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
#lister['chatgpt description'] = \
lister_fulltrain['law short processed'] = lister_fulltrain['law short2'].map(lambda x: x.lower())
# Print out the first rows
lister_fulltrain['law short processed'].head()


# In[58]:


#factorize company field - new columnn with the Manual industries coded into numbers representing each
lister_fulltrain['Industry Number'] = pd.factorize(lister_fulltrain['Company Field'])[0]
#view first 5 rows
#lister_fulltrain.head()


# In[59]:


#replace black or NA values in the description column
fill = "no description blank company or individual"

lister_fulltrain['chatgpt description processed'].fillna(fill, inplace = True)
lister_fulltrain['chatgpt description'].fillna(fill, inplace = True)
lister_fulltrain = lister_fulltrain.dropna(subset=['chatgpt description processed'])


# In[60]:


#save description column on its own and view
text = lister_fulltrain['chatgpt description processed']
text.head


# In[61]:


#convert the text column variable into a list and remove None values
train_documents = text.values.tolist()
train_documents = list(filter(lambda x: x is not None, train_documents))
#train_documents 


# In[62]:


#create the documents for the clustering as the train_documents list
documents = train_documents


# # Kmeans Clustering Algorithm with Text Inputs

# In[63]:


#initialize the text vectorizer with englush stop words: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
#set inputs X as the vectorized documents
X = vectorizer.fit_transform(documents)


# Selecting the Amount of Clusters, k, with the Elbow Method
# * An optimal selection is usually when the line bends and the point become closer together
# * There is no set correct amount, different selections can be made and run to see what the best fit will be
# * The amount of k will vary on the task and data used. 
# * This Analysis: k=10 was chosen. optimal k lies beteew about 5-10 and input from the manual anlaysis was also taken into consideration of the amount of groups manually seen for the amount of k to choose. This way we coul daccount for other possible subindusties amoung the data. 

# In[64]:


#run clusters on many k selections with the X inputs
Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)


# In[14]:


#output the elbow method plot
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title("elbow method for optimal k")
plt.show()
#optimal k lies beteew about 5-10


# Running the Clustering Model

# In[83]:


#fit model to the training data: text documents X
k = 10 # edit k here to explore results of different k selections
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


# In[91]:


#model fits all documents into a cluster
#save the cluster assignments as a new column in the dataset
lister_fulltrain['cluster'] = model.labels_


# In[92]:


#view the first 5 rows, cluster numbers should be on the end now
lister_fulltrain.head()


# # Visualizations of the Documents (Text Descriptions)

# In[18]:


#creat a list of stop words to not include in the visualizations
#these are fillers or words that are very common and displayed to much over the importatnt words we care about
stop = ['and', 'other','to','area','areas','possible','specialize', "office", 'specializes', 'is', 'not', 'a', 'that', 'or', 'in', 'of', 'including', 'answer','this']


# Word Cloud for All Company Descriptions

# In[138]:


# Join the different processed titles together.
long_string = ','.join(list(lister_fulltrain['chatgpt description'].values))
# Create a WordCloud object - all inputs can be edited for a different visual result
wordcloud = WordCloud(background_color="white", 
                      max_words=5000, 
                      contour_width=3, 
                      contour_color='steelblue', 
                      stopwords=stop)
# Generate a word cloud FOR ALL DESCRIPTIONS
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# Word Cloud of All Law Descritptions

# In[145]:


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(lister_fulltrain['law description'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', stopwords=stop)
# Generate a word cloud FOR ALL LAW DESCRIPTIONS
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# All Law Descriptions Word Cloud Redone Excludig Filler/Unessesary Words ie Stop Words

# In[19]:


#craete another list of stops words to remove from the crowded law wordcloud
stop2 = ['and', 'other','to','area','areas','possible','specialize', "firm", "office", 'law', 'specializes', 'is', 'not', 'a', 'that', 'or', 'in', 'of', 'including', 'answer','this']


# In[154]:


wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',stopwords= stop2).generate(' '.join(lister_fulltrain['law description']))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Short Law Description Word Cloud

# In[173]:


stop3 = ['and', 'corporate','planning','tax','other',"intellectual","property",'estate','to','area','areas','possible','specialize', "firm", "office", 'law', 'specializes', 'is', 'not', 'a', 'that', 'or', 'in', 'of', 'including', 'answer','this']
#SHORT LAW DESCRIPTIONS law word cloud redone excludig filler/unessesary words ie stop words
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',stopwords= stop3).generate(' '.join(lister_fulltrain['law short processed']))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # Visualizing the Clusters

# Top Terms from the Descriptions within Each Cluster 

# In[93]:


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# Function to print out word clouds for each cluster but will capture more filler words than the top terms list above. 
# Can initialize a list of stop words like earlier to output visuals without

# In[88]:


def word_cloud(text, wc_title):
    # Create stopword list
    stopword_list = set(STOPWORDS) 

    # Create WordCloud 
    word_cloud = WordCloud(width = 800, height = 500, 
                           background_color ='white', 
                           stopwords = stopword_list, 
                           min_font_size = 14).generate(text) 

    # Set wordcloud figure size
    plt.figure(figsize = (8, 6)) 
    
    # Set title for word cloud
    plt.title(wc_title)
    
    # Show image
    plt.imshow(word_cloud) 

    # Remove Axis
    plt.axis("off")  

    # save word cloud
    #plt.savefig(wc_file_name,bbox_inches='tight')

    # show plot
    plt.show()


# In[94]:


#output of function to group text description in each cluster label
lister=pd.DataFrame({"text":lister_fulltrain['chatgpt description processed'],"labels":lister_fulltrain['cluster']})


for i in lister.labels.unique():
    new_lister=lister[lister.labels==i]
    text="".join(new_lister.text.tolist())
    word_cloud(text, lister.labels.unique()[i]))


# # Name the Clusters and Map Names to the Cluster Numbers

# In[26]:


#total unique clusters
lister.labels.unique()


# In[101]:


#map cluster names by naming the clusters based on key words and word clouds
current_labels = lister.labels.unique()
desired_labels = ('Law','Financial Services','Other', 'Insurance', 'Banking - Trust',
    'Law', ' Law - Family', 'Banking', 'Financial Services - Tax & Accounting' ,'Government')
# create a dictionary for your corresponding values
map_dict = dict(zip(current_labels, desired_labels))
map_dict= {0: 'Law', 1: 'Financial Services - Wealth & Invest', 2: 'Other', 3: 'Insurance', 4: 'Banking - Trust',
    5: 'Law', 6:' Law - Family', 7:'Banking & Finance', 8:'Financial Services - Tax & Accounting' ,9:'Government'}

# map the desired values back to the dataframe
# this will replace the original values
#lister_fulltrain['cluster'] = lister_fulltrain['cluster'].map(map_dict)

#map to a new column if you want to preserve the old values
lister_fulltrain['cluster names'] = lister_fulltrain['cluster'].map(map_dict)


# In[ ]:


# manual industry fields assigned for reference
#Banking
#Financial Services
#Blank/Individual
#Law
#Insurance
#Business & Estate Planning
#Government 
#Nonprofit
#Manufacturing/Industrial
#Real Estate
#Technology/IT
#Healthcare
#Education
#Insurance 


# In[102]:


lister_fulltrain.head()


# In[ ]:





# # Clustering On New Data

# import new data you would like to fit into the clusters from the model and preprocess text fields

# In[133]:


data2 = pd.read_excel(r'C:\\ .xlsx')
#drop range of columns not needed
data2 = data2.drop(data2.loc[:, 'Law':'Technology/IT'].columns, axis=1)
#drop rows based on a specific value in one column
data2.drop(data2.index[data2['Company Field'] == 'Blank/Individual'], inplace = True)
#print out infomrmation on all columns now
data2.info()


# In[134]:


#save data as a new set
lister_pred = data2
# change variable types if needed
#lister_pred["chatgpt description"] = lister_pred["chatgpt description"].astype(str)
#lister_pred["company"] = lister_pred["company"].astype(str)
#lister_pred['law description']=lister_pred['law description'].astype(str)
#lister_pred['law description short']=lister_pred['law description short'].astype(str)
print(lister_pred.dtypes) #need to display as objects below

#preprocess the description text
# Remove punctuation
lister_pred['chatgpt description2'] = ""
lister_pred['chatgpt description2'] = lister_pred['chatgpt description'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
#lister['chatgpt description'] = \
lister_pred['chatgpt description processed'] = lister_pred['chatgpt description2'].map(lambda x: x.lower())
# Print out the first rows
lister_pred['chatgpt description processed'].head()


# In[135]:


# Prediction short law description text
# Remove punctuation
lister_pred['law short2'] = ""
lister_pred['law short2'] = lister_pred['law description short'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
#lister['chatgpt description'] = \
lister_pred['law short processed'] = lister_pred['law short2'].map(lambda x: x.lower())
# Print out the first rows
lister_pred['law short processed'].head()


# In[136]:


#make y prediction column
lister_pred['Industry Number'] = ""
lister_pred.head()


# Create the Documents and Model Inputs

# In[137]:


#save description as thier own variable, convert to a list, remove None values
text2 = lister_pred['chatgpt description processed']
text2.head
test_documents = text2.values.tolist()
test_documents = list(filter(lambda x: x is not None, test_documents))
#view the documents
#test_documents


# In[139]:


#save the text as another set of documents
documents2 = test_documents


# In[140]:


#vectorize the X input from the prediction data documents
#do not fit the vectorizer again, only need vectorizer.transform()
X2 = vectorizer.transform(documents2)


# Predict Clusters on New Data Documents and Map Cluster Names to the Labels

# In[141]:


#pred_labels are the clusters it assigns/predicts for the new data based on the training model already fit above
pred_labels = model.predict(X2) # test is your data to predict the cluster


# In[142]:


#create column with the cluster labels for each row
lister_pred['cluster'] = pred_labels
#lister_pred.head()


# In[144]:


#create column for cluster names to match the names mapped to the clusters earlier
lister_pred['cluster names'] = lister_pred['cluster'].map(map_dict)


# In[145]:


#view the clusters
lister_pred.head()


# # Export the Data Sets to Further Explore the Clusters

# In[146]:


lister_fulltrain.to_csv(r'C:\\ .txt', index = None, sep = ',', mode = 'w')
lister_pred.to_csv(r'C:\\  .txt', index = None, sep = ',', mode = 'w')


# In[ ]:




