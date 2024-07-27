import unittest
import pandas as pd
from src.customer_segmentation import load_data, clean_data, perform_eda

class TestCustomerSegmentation(unittest.TestCase):
    
    def setUp(self):
        self.file_path = 'C:/Users/yadav/OneDrive/Desktop/Module Proj/Customer Segmentation/Mall_Customers.csv'
        self.df = load_data(self.file_path)
    
    def test_load_data(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(self.df.shape[1], 5)
    
    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        self.assertFalse(cleaned_df.isnull().any().any())
    

if __name__ == '__main__':
    unittest.main()
