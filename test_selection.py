import unittest
from chronic.selection import column_name,column_number,select_features


class SelectionTests(unittest.TestCase):
    def test_console_column_number_requires_index(self):
        self.assertEqual(column_number('2',['Age','class']),'class')
        with self.assertRaises(ValueError):column_number('Age',['Age','class'])

    def setUp(self):
        self.schema={'target':'disease','numeric':['age','bmi'],'categorical':['sex'],
                     'allow_missing':['bmi'],'ranges':{'age':[0,120],'bmi':[10,80]}}

    def test_indices_are_one_based_and_names_supported(self):
        self.assertEqual(column_name('2',['age','bmi','disease']),'bmi')
        self.assertEqual(column_name('disease',['age','bmi','disease']),'disease')
        with self.assertRaises(ValueError):column_name('0',['age'])

    def test_subset_preserves_original_and_filters_constraints(self):
        result=select_features(self.schema,['age','sex'])
        self.assertEqual(result['numeric'],['age'])
        self.assertEqual(result['categorical'],['sex'])
        self.assertEqual(result['allow_missing'],[])
        self.assertEqual(result['ranges'],{'age':[0,120]})
        self.assertEqual(self.schema['numeric'],['age','bmi'])

    def test_target_unknown_and_duplicate_columns_rejected(self):
        for names in [['disease'],['patient_id'],['age','age'],[]]:
            with self.assertRaises(ValueError):select_features(self.schema,names)


if __name__=='__main__':unittest.main()
