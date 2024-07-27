import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

def load_data(file_path):
    warnings.filterwarnings('ignore')
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        df.dropna(inplace=True)
        print("\nRows with missing values are dropped.")
    else:
        print("\nNo missing values found.")
    return df

def perform_eda(df):
    # EDA based on gender
    sns.countplot(x='Gender', data=df)
    plt.title('Count Plot of Gender')
    plt.show()

    # EDA based on Age
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True)
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.show()

    # EDA based on Annual Income
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Annual Income (k$)'], kde=True)
    plt.title('Histogram of Annual Income')
    plt.xlabel('Annual Income (k$)')
    plt.show()

    # Count of male and female w.r.t Age
    plt.figure(figsize=(20, 7))
    sns.countplot(data=df, x="Age", hue="Gender")
    plt.title('Count of Male and Female by Age')
    plt.xlabel('Age')
    plt.show()

def kmeans_clustering_Income_Spend(df):
    df1 = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scatter plot
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df1)
    plt.title('Scatter Plot of Annual Income vs Spending Score')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    # Elbow Method for Optimal Number of Clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df1)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans_model = KMeans(n_clusters=5, random_state=42)
    y_kmeans = kmeans_model.fit_predict(df1)
    df1['Cluster'] = y_kmeans

    # Scatter plot for clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df1, palette='Set1', legend='full')
    plt.title('Clusters of customers (K-Means Clustering)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df1)

    # K-means clustering on PCA-transformed data
    kmeans_model_pca = KMeans(n_clusters=5, random_state=42)
    y_kmeans_pca = kmeans_model_pca.fit_predict(X_pca)
    df1['Cluster PCA'] = y_kmeans_pca

    # Scatter plot for clusters after PCA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans_pca, palette='Set1', legend='full')
    plt.scatter(kmeans_model_pca.cluster_centers_[:, 0], kmeans_model_pca.cluster_centers_[:, 1], s=100, c='black', marker='+', label='Cluster Centers')
    plt.title('Clusters of customers after PCA (K-Means Clustering)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return df1

def kmeans_clustering_age_income(df):
    df2 = df[['Age', 'Annual Income (k$)']]
    
    # Scatter plot
    sns.scatterplot(x='Age', y='Annual Income (k$)', data=df2)
    plt.title('Scatter Plot of Age vs Annual Income')
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
    plt.show()

    # Elbow Method for Optimal Number of Clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df2)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Applying K-means clustering with optimal clusters
    kmeans_model = KMeans(n_clusters=4, random_state=42)
    y_kmeans = kmeans_model.fit_predict(df2)
    df2['Cluster'] = y_kmeans

    # Scatter plot for clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Annual Income (k$)', hue='Cluster', data=df2, palette='Set1', legend='full')
    plt.title('Clusters of customers (K-Means Clustering)')
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
    plt.show()

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df2)

    # K-means clustering on PCA-transformed data
    kmeans_model_pca = KMeans(n_clusters=4, random_state=42)
    y_kmeans_pca = kmeans_model_pca.fit_predict(X_pca)
    df2['Cluster PCA'] = y_kmeans_pca

    # Scatter plot for clusters after PCA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans_pca, palette='Set1', legend='full')
    plt.scatter(kmeans_model_pca.cluster_centers_[:, 0], kmeans_model_pca.cluster_centers_[:, 1], s=100, c='black', marker='+', label='Cluster Centers')
    plt.title('Clusters of customers after PCA (K-Means Clustering)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return df2

def kmeans_clustering_age_spending(df):
    df3 = df[['Age', 'Spending Score (1-100)']]
    
    # Scatter plot
    sns.scatterplot(x='Age', y='Spending Score (1-100)', data=df3)
    plt.title('Scatter Plot of Age vs Spending Score (1-100)')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    # Elbow Method for Optimal Number of Clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df3)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Applying K-means clustering with optimal clusters
    kmeans_model = KMeans(n_clusters=4, random_state=42)
    y_kmeans = kmeans_model.fit_predict(df3)
    df3['Cluster'] = y_kmeans

    # Scatter plot for clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', data=df3, palette='Set1', legend='full')
    plt.title('Clusters of customers (K-Means Clustering)')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df3)

    # K-means clustering on PCA-transformed data
    kmeans_model_pca = KMeans(n_clusters=4, random_state=42)
    y_kmeans_pca = kmeans_model_pca.fit_predict(X_pca)
    df3['Cluster PCA'] = y_kmeans_pca

    # Scatter plot for clusters after PCA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans_pca, palette='Set1', legend='full')
    plt.scatter(kmeans_model_pca.cluster_centers_[:, 0], kmeans_model_pca.cluster_centers_[:, 1], s=100, c='black', marker='+', label='Cluster Centers')
    plt.title('Clusters of customers after PCA (K-Means Clustering)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return df3

# Main Function

def main():
    file_path = 'C:/Users/yadav/OneDrive/Desktop/Module Proj/Customer Segmentation/Mall_Customers.csv'
    
    # Load data
    df = load_data(file_path)
    print(f'Data: \n{df.head()}')
    
    
    # Data Cleaning
    cleaned_df = clean_data(df)
    print(f'\nStatistical Information of Cleaned Data: \n{cleaned_df.describe()}')
    
    
    # EDA
    perform_eda(cleaned_df)
    
    
    # Applying Elbow Method and Clustering for Income & Spending
    df_income_spending = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    print(f'\nApplying elbow method on below data: \n{df_income_spending.head()}')
    
    wcss_income_spending = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df_income_spending)
        wcss_income_spending.append(kmeans.inertia_)
    
    for i, w in enumerate(wcss_income_spending, 1):
        print(f'WCSS for {i} clusters (Annual Income & Spending Score): {w}')
    
    # Clustering for Income & Spending
    clustered_income_spending = kmeans_clustering_Income_Spend(df_income_spending)
    print("\nClustered data for Annual Income & Spending Score:")
    print(clustered_income_spending.head())
    
    
    # Applying Elbow Method and Clustering for Age & Annual Income
    df_age_income = df[['Age', 'Annual Income (k$)']]
    print(f'\nApplying elbow method on below data: \n{df_age_income.head()}')
    
    wcss_age_income = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df_age_income)
        wcss_age_income.append(kmeans.inertia_)
    
    for i, w in enumerate(wcss_age_income, 1):
        print(f'WCSS for {i} clusters (Age & Annual Income): {w}')
    
    # Clustering for Age & Income
    clustered_age_income = kmeans_clustering_age_income(df_age_income)
    print("\nClustered data for Age & Annual Income:")
    print(clustered_age_income.head())
    
    
    # Applying Elbow Method and Clustering for Age & Spending
    df_age_spending = df[['Age', 'Spending Score (1-100)']]
    print(f'\nApplying elbow method on below data: \n{df_age_spending.head()}')
    
    wcss_age_spending = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df_age_spending)
        wcss_age_spending.append(kmeans.inertia_)
    
    for i, w in enumerate(wcss_age_spending, 1):
        print(f'WCSS for {i} clusters (Age & Spending Score): {w}')
    
    # Clustering for Age & Spending
    clustered_age_spending = kmeans_clustering_age_spending(df_age_spending)
    print("\nClustered data for Age & Spending Score:")
    print(clustered_age_spending.head())
    
    
if __name__ == "__main__":
    main()
