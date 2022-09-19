
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import warnings
warnings.filterwarnings("ignore") 

st.title('Mercedes Benz Greener Manufacturing : Predicting the Test Time of a Mercedes Benz', anchor=None)

image = Image.open('Mercedes_Logo_11.jpg')
st.image(image, caption='Mercedes Benz')

st.header('Objective')
st.write('Our objective is to predict the testing time of a Mercedes Benz spent on a test bench. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium car makers. Cars when manufactured cannot be put straight away to run on the road without testing. Every vehicle has to go some testing parameters before they hit the road for our use. Testing is a very crucial step in any automotive industry and all the car manufacturers do the testing of their cars before entering to the market so as to maintain the safety of the passengers in the vehicle. So, a premium and popular brand in automotive industry like Mercedes Benz does not compromise on the quality and testing of their vehicles and hence with this problem they want to minimize the testing time that each vehicle spent on the testing so as to reduce the testing cost and CO2 emissions also. The goal of this study is to have a robust and efficient testing system such that the testing time is reduced without compromising the quality')

@st.cache
def load_data(nrows):
    data_train = pd.read_csv('train.csv', nrows=nrows)
    data_test = pd.read_csv('test.csv', nrows=nrows)

    return data_train,data_test

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data_train,data_test = load_data(4209)
y = data_train['y']
rows = len(data_train)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading Data...Done! (using st.cache)")

if st.checkbox('Show Raw Data'):
    st.subheader('Raw data')
    st.write(data_train)
    
st.write('Now as we have seen the crux of the raw data we know that each vehicle has a set of Categorical features and Binary features in data and by these features we have to determine the impact of the features on our ground truth value(ie. Our testing time(y) in raw data)')
    
st.header('Analysing Ground Truth Values')

def main():
    page = st.sidebar.selectbox(
        "Select a Page",
        [
            "Violin , Distribution and Scatter Plot of Test Time" #New Page
        ]
    )
    violinDist_plot()
    
def violinDist_plot():
    st.subheader("Violin, Distribution and Scatter Plot")
    sd = st.selectbox(
        "Select a Plot", #Drop Down Menu Name
        [
            "Violin Plot", #First option in menu
            "Distribution Plot", #Second option in menu
            "Scatter Plot" #Third option in menu
        ]
    )

    fig = plt.figure(figsize=(12, 6))

    if sd == "Violin Plot":
        sns.violinplot(data = data_train['y'])
        plt.xlabel('Y')
        plt.ylabel('Time (secs)')
    
    elif sd == "Distribution Plot":
        sns.distplot(data_train['y'])
        plt.xlabel('y')
        plt.ylabel('Density')
        
    elif sd == "Scatter Plot":
        plt.scatter(range(len(data_train['y'])), data_train['y'])
        plt.xlabel("Index")
        plt.ylabel("Testing Time (Seconds)")
        
    st.pyplot(fig)

if __name__ == "__main__":
    main()
    
st.write('Above plots shows us that there are outliers present in the ground truth values')

st.header('Analysing Categorical features')

def main():
    
    page = st.sidebar.selectbox(
        "Select a Page",
        [
            "Analysis of Categorical Features" #New Page
        ]
    )
    analyze_cat()
    
def analyze_cat():
    st.header("Categorical Features")
    sd = st.selectbox(
        "Select a Feature", #Drop Down Menu Name
        [
           'X0','X1','X2','X3','X4','X5','X6','X8'
        ]
    )
    fig = plt.figure(figsize=(12, 6))

    if sd == "X0":
        sns.boxplot(x=data_train['X0'],y = data_train['y'])
        plt.xlabel('X0', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)
    
    elif sd == "X1":
        sns.boxplot(x=data_train['X1'],y = data_train['y'])
        plt.xlabel('X1', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)
        
    elif sd == "X2":
        sns.boxplot(x=data_train['X2'],y = data_train['y'])
        plt.xlabel('X2', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)
        
    elif sd == "X3":
        sns.boxplot(x=data_train['X3'],y = data_train['y'])
        plt.xlabel('X3', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)  
        
    elif sd == "X4":
        sns.boxplot(x=data_train['X4'],y = data_train['y'])
        plt.xlabel('X4', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)   
        
    elif sd == "X5":
        sns.boxplot(x=data_train['X5'],y = data_train['y'])
        plt.xlabel('X5', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)  
        
    elif sd == "X6":
        sns.boxplot(x=data_train['X6'],y = data_train['y'])
        plt.xlabel('X6', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)    
        
    elif sd == "X8":
        sns.boxplot(x=data_train['X8'],y = data_train['y'])
        plt.xlabel('X8', fontsize=13)
        plt.ylabel('Time', fontsize=13)
        #ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.tick_params(axis='both', which='minor', labelsize=15)        
        
    st.pyplot(fig)
    
    st.subheader('Observation')
    
    st.write('A. When we observe the box plots of X0 we can clearly see that (az) and (bc) are two customizations that are not overlapping with the other customizations and hence they can be grouped as seperate entities.\n 1. Customizations like (s,n,f and y) and (b , u, ad) are all having the same distribution of time. So we can infer from this that maybe the customizations are different in their behaviours but may take same test time.\n  2. Some customizations like (k) have very large distribution of test time and it concludes that these type of customizations are crucial and they may take large time or may end up early according to the vehicle model.\n B. In X1 and X5 all the customizations have almost same distribution except (v) in X1 and (x) in X5. X6 and X8 have same distributed customizations. X4 has very less variance and it can be removed as it is not providing that much of the information.')

if __name__ == "__main__":
    main()

st.header('Analysing Binary features')

def main():
    
    page = st.sidebar.selectbox(
        "Select a Page",
        [
            "Analysis of Binary Features" #New Page
        ]
    )
    analyze_bin()
    
def analyze_bin():
    st.header("Binary Features")
    sd = st.selectbox(
        "Select a plot for Feature", #Drop Down Menu Name
        [
          'Feature Importance','Mean Values of Binary Variables'
        ]
    )
    fig = plt.figure(figsize=(12, 6))
    
    cat_features = ['X0','X1','X2','X3','X4','X5','X6','X8']
    binary_features = data_train.columns.drop(['ID', 'y'] + list(cat_features))
    
    if sd == "Feature Importance":
        binary_dataframe = data_train.drop(['ID','y','X0', 'X1', 'X2', 'X4','X3', 'X5', 'X6', 'X8','X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293','X297', 'X330', 'X339', 'X347','X35', 'X37', 'X39', 'X54', 'X76', 'X84', 'X90', 'X94', 'X102', 'X113', 'X119', 'X122', 'X129', 'X134', 'X137', 'X140', 'X146', 'X147', 'X172', 'X198', 'X199', 'X213', 'X214', 'X215', 'X216', 'X217', 'X222', 'X226', 'X227', 'X232', 'X239', 'X242', 'X243', 'X244', 'X245', 'X247', 'X248', 'X249', 'X250', 'X253', 'X254', 'X262', 'X263', 'X264', 'X266', 'X279', 'X296', 'X299', 'X302', 'X320', 'X324', 'X326', 'X348', 'X352', 'X360', 'X363', 'X364', 'X365', 'X368', 'X370', 'X371', 'X382', 'X385'],axis=1)
        result_y = data_train['y']

        #create a Regressor model
        rf = joblib.load('feature_imp_rf.pkl')
        # get the sorted indices of features
        features = binary_dataframe.columns
        importances = rf.feature_importances_
        imp_feat_idx = np.argsort(importances)[-10:]

        plt.title('Feature Importances')
        plt.barh(range(len(imp_feat_idx)), importances[imp_feat_idx], color='r', align='center')
        plt.yticks(range(len(imp_feat_idx)), [features[i] for i in imp_feat_idx],fontsize=15)
        plt.xlabel('Relative Importance')
        
    elif sd == "Mean Values of Binary Variables":
        binary_means = [np.mean(data_train[c]) for c in binary_features]
        binary_names = np.array(binary_features)[np.argsort(binary_means)]
        binary_means = np.sort(binary_means)
    
        fig, ax = plt.subplots(1, 3, figsize=(15,30))
        ax[0].set_ylabel('Feature name')
        ax[1].set_title('Mean values of binary variables')
        for i in range(3):
            names, means = binary_names[i*119:(i+1)*119], binary_means[i*119:(i+1)*119]
            ax[i].barh(range(len(means)), means, color='green')
            ax[i].set_xlabel('Mean value')
            ax[i].set_yticks(range(len(means)))
            ax[i].set_yticklabels(names, rotation='horizontal')

    st.pyplot(fig)
    st.write('We can observe from the Feature Importance plot that X314 is the most important feature followed by X315 and X118')

if __name__ == "__main__":
    main()
############################################################################################################################################    
def clean_categorical(data):
    
    """
    This function takes the dataframe as input and
    encodes the categorical features.
    """
    # create empty lists for collecting feature names
    cat_features = ['X0','X1','X2','X3','X4','X5','X6','X8']
    
    Binary_features = ['X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 
                       'X24', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                       'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 
                       'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 
                       'X70', 'X71', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85',
                       'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100',
                       'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107', 'X108', 'X109', 'X110', 'X111', 'X112', 'X113',
                       'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120', 'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 
                       'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134', 'X135', 'X136', 'X137', 'X138', 'X139', 'X140',
                       'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147', 'X148', 'X150', 'X151', 'X152', 'X153', 'X154', 
                       'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161', 'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 
                       'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 
                       'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187', 'X189', 'X190', 'X191', 'X192', 'X194', 'X195', 
                       'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 
                       'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215', 'X216', 'X217', 'X218', 'X219', 'X220', 'X221',
                       'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228', 'X229', 'X230', 'X231', 'X232', 'X233', 'X234', 
                       'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241', 'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 
                       'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254', 'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 
                       'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267', 'X268', 'X269', 'X270', 'X271', 'X272', 'X273', 
                       'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280', 'X281', 'X282', 'X283', 'X284', 'X285', 'X286',
                       'X287', 'X288', 'X289', 'X290', 'X291', 'X292', 'X293', 'X294', 'X295', 'X296', 'X297', 'X298', 'X299',
                       'X300', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307', 'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 
                       'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320', 'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 
                       'X327', 'X328', 'X329', 'X330', 'X331', 'X332', 'X333', 'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 
                       'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346', 'X347', 'X348', 'X349', 'X350', 'X351', 'X352',
                       'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359', 'X360', 'X361', 'X362', 'X363', 'X364', 'X365',
                       'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372', 'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 
                       'X379', 'X380', 'X382', 'X383', 'X384', 'X385']
    
    
    # create categorical feature dataframe
    cat_df = data[cat_features]
    oe = OrdinalEncoder()
    enc_cat_df = oe.fit_transform(cat_df)
    enc_df = pd.DataFrame(enc_cat_df,columns=cat_features)
    enc_df.insert(0, 'ID', data['ID'].values)
    binary_df = data.drop(['X0', 'X1', 'X2', 'X3', 'X4','X5', 'X6', 'X8'],axis=1)
    for col in data.columns:
        if col=='y':
            binary_df = binary_df.drop(['y'],axis=1)
            break
            
    final_df = pd.merge(enc_df,binary_df,on='ID',how='left')

    return final_df.drop(['ID'],axis=1)

################################################################################################################################

# creating a final prediction function

def final_fun_1(X):
    
    # preprocess the data
    X_processed = clean_categorical(X)
    #load the saved model 1
    model1 = joblib.load("final_best_model1_xgb.pkl")
    # predict the targets
    y_pred1 = model1.predict(X_processed)
    dtest = xgb.DMatrix(X_processed)
    #load the saved model 2
    model2 = joblib.load("final_best_model2_xgb.pkl")
    # predict the targets
    y_pred2 = model2.predict(dtest)
    # Average the test data preditions of both models
    avg_pred = (y_pred1 + y_pred2)/2
    
    model3 = joblib.load("final_best_model3_stacked.pkl")
    y_pred3 = model3.predict(X_processed)
    
    #final prediction: taking averaged models of 1 and 2 as output and aveaging the output with stacked model output
    final_prediction = (avg_pred + y_pred3)/2
    
    return final_prediction

################################################################################################################################


def final_fun_2(X,y):
    # preprocess the data
    X_processed = clean_categorical(X)
    # load the saved model 1
    model1 = joblib.load("final_best_model1_xgb.pkl")
    # predict the targets
    y_pred1 = model1.predict(X_processed)
    dtest = xgb.DMatrix(X_processed)
    #load the saved model 2
    model2 = joblib.load("final_best_model2_xgb.pkl")
    # predict the targets
    y_pred2 = model2.predict(dtest)
    # Average the test data preditions of both models
    avg_pred = (y_pred1 + y_pred2)/2
    #load the saved model 3
    model3 = joblib.load("final_best_model3_stacked.pkl")
    y_pred3 = model3.predict(X_processed)
    
    #final prediction: taking averaged models of 1 and 2 as output and aveaging the output with stacked model output
    final_prediction = (avg_pred + y_pred3)/2
    
    return r2_score(y,final_prediction)

st.header('Predict the test time for a Mercedes Car in a test bench')
filter = st.slider('How many prediction do you want?', min_value=1, max_value= 50)
filtered_data_test = data_test[:filter]

if st.button('Predict Test Time'):
    predict_time = final_fun_1(filtered_data_test)
    for idx,pred in enumerate(predict_time):
        st.success(f'{idx+1}. Predicted test time is : {pred}')

st.subheader('R2 Score of the model is')
r2_score = final_fun_2(data_train,y)
st.success(f'The R2 Score is : {r2_score}')

st.sidebar.markdown('<a href="https://github.com/RetroX6">GitHub</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://www.linkedin.com/in/gaurav-thakur-055785159/">LinkedIn</a>', unsafe_allow_html=True)
