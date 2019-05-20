import unittest
import pandas as pd
import numpy as np
from htsprophet.hts import hts, makeWeekly, orderHier


class testHTSOut(unittest.TestCase):
    
    def testOutputs(self):
        date = pd.date_range("2014-04-02", "2017-07-17")
        medium = ["Air", "Land", "Sea"]
        businessMarket = ["Birmingham","Auburn","Evanston"]
        platform = ["Stone Tablet","Car Phone"]
        mediumDat = np.random.choice(medium, len(date))
        busDat = np.random.choice(businessMarket, len(date))
        platDat = np.random.choice(platform, len(date))
        sessions = np.random.randint(100,40000,size=(len(date),1))
        data = pd.DataFrame(date, columns = ["day"])
        data["medium"] = mediumDat
        data["platform"] = platDat
        data["businessMarket"] = busDat
        data["sessions"] = sessions
        data1 = makeWeekly(data)
        data2, nodes = orderHier(data1, 1, 2, 3)
        ##
        # Check Make Weekly
        ##
        self.assertIsNotNone(data1)
        self.assertLessEqual(len(data1.iloc[:,0]), len(data.iloc[:,0]))
        ##
        # Check OrderHier
        ##
        self.assertIsNotNone(data2)
        self.assertIsNotNone(nodes)
        self.assertLessEqual(len(data2.iloc[:,0]), len(data1.iloc[:,0]))
        self.assertEqual(len(data2.iloc[:,0]), len(data2.iloc[:,0].unique()))
        ##
        # Test if bottom up is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "BU")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for bottom up is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if average historical proportions is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "AHP")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for average historical proportions is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if proportion of historical averages is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "PHA")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for proportions of historical averages is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if forecast proportions is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "FP")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for forecast proportions is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if ordinary least squares is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "OLS")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for ordinary least squares is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if structurally weighted least squares is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "WLSS")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for structurally weighted least squares is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if variance weighted least squares is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "WLSV")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for variance weighted least squares is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test if cross validation is working correctly
        ##
        myDict = hts(data2, 52, nodes, freq = 'W', method = "cvSelect")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for cross validation is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Test to see if BoxCox transform changed the data at all
        ##
        data3 = data2
        myDict = hts(data2, 52, nodes, freq = 'W', method = "cvSelect", transform = "BoxCox")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        for i in range(len(data2.columns.tolist())-1):
            self.assertEqual(data2.iloc[:,i+1].all(), data3.iloc[:,i+1].all())
        ##
        # Test if Logistic data is working correctly
        ##
        data2.iloc[:,1:] = np.log(data2.iloc[:,1:]+1)
        myDict = hts(data2, 52, nodes, freq = 'W', method = "FP", cap = 11, capF = 13)
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        ##
        # Check the names of the dictionary
        ##
        for key in myDict.keys():
            self.assertTrue(isinstance(key, str))
        self.assertEqual(len(myDict['Total'].yhat), data2.shape[0]+52)
        ##
        # Test if the aggregation result for cross validation is almost equal
        ##
        self.assertAlmostEqual(myDict['Total'].yhat[-52:].all(), (myDict['Air'].yhat[-52:] + myDict['Land'].yhat[-52:] + myDict['Sea'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Air'].yhat[-52:].all(), (myDict['Air_Stone Tablet'].yhat[-52:] + myDict['Air_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Land'].yhat[-52:].all(), (myDict['Land_Stone Tablet'].yhat[-52:] + myDict['Land_Car Phone'].yhat[-52:]).all())
        self.assertAlmostEqual(myDict['Sea'].yhat[-52:].all(), (myDict['Sea_Stone Tablet'].yhat[-52:] + myDict['Sea_Car Phone'].yhat[-52:]).all())
        ##
        # Testing for system exit
        ##
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 52, nodes, freq = 'W', method = "Yellow", cap = 11, capF = 13)
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 0, nodes, freq = 'W', method = "FP", cap = 11, capF = 13)
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 52, [[72],[8]], freq = 'W', method = "FP", cap = 11, capF = 13)
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 52, nodes, freq = 'W', method = "FP", cap = 'mycap', capF = 13)
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 52, nodes, freq = 'W', method = "FP", cap = 11, capF = 'mycap')
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 52, nodes, freq = 'W', method = "FP", cap = pd.DataFrame(nodes), capF = 13)
        with self.assertRaises(SystemExit):
            myDict = hts(data2, 52, nodes, freq = 'W', method = "FP", cap = 11, capF = pd.DataFrame(nodes))
        
if __name__ == '__main__':
    unittest.main()