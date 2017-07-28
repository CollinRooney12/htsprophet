import unittest
import pandas as pd
import numpy as np
from htsprophet.hts import hts
from htsprophet.runHTS import orderHier, makeWeekly 


class testHTSOut(unittest.TestCase):
    
    def testOutputs(self):
        date = pd.date_range("2014-04-02", "2017-07-17")
        medium = ["Air", "Land", "Sea"]
        businessMarket = ["Birmingham","Auburn","Evanston"]
        platform = ["Stone Tablet","Car Phone"]
        mediumDat = np.random.choice(medium, len(date))
        busDat = np.random.choice(businessMarket, len(date))
        platDat = np.random.choice(platform, len(date))
        sessions = np.random.randint(10,1000,size=(len(date),1))
        data = pd.DataFrame(date, columns = ["day"])
        data["medium"] = mediumDat
        data["platform"] = platDat
        data["businessMarket"] = busDat
        data["sessions"] = sessions
        data1 = makeWeekly(data)
        data2, nodes = orderHier(data1, 1, 2, 3)
        myDict = hts(data2, 52, nodes, freq = 'W', method = "BU")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict[0].yhat), data2.shape[0]+52)
        self.assertTrue(all(myDict[0].yhat <= (myDict[1].yhat + myDict[2].yhat + myDict[3].yhat)*1.001))
        myDict = hts(data2, 52, nodes, freq = 'W', method = "AHP")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict[0].yhat), data2.shape[0]+52)
        self.assertTrue(all(myDict[0].yhat <= (myDict[1].yhat + myDict[2].yhat + myDict[3].yhat)*1.001))
        myDict = hts(data2, 52, nodes, freq = 'W', method = "PHA")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict[0].yhat), data2.shape[0]+52)
        self.assertTrue(all(myDict[0].yhat <= (myDict[1].yhat + myDict[2].yhat + myDict[3].yhat)*1.001))
        myDict = hts(data2, 52, nodes, freq = 'W', method = "FP")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict[0].yhat), data2.shape[0]+52)
        self.assertTrue(all(myDict[0].yhat <= (myDict[1].yhat + myDict[2].yhat + myDict[3].yhat)*1.001))
        myDict = hts(data2, 52, nodes, freq = 'W', method = "OC")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict[0].yhat), data2.shape[0]+52)
        self.assertTrue(all(myDict[0].yhat <= (myDict[1].yhat + myDict[2].yhat + myDict[3].yhat)*1.001))
        myDict = hts(data2, 52, nodes, freq = 'W', method = "cvSelect")
        self.assertIsNotNone(myDict)
        self.assertEqual(len(myDict), sum(map(sum, nodes))+1)
        self.assertEqual(len(myDict[0].yhat), data2.shape[0]+52)
        self.assertTrue(all(myDict[0].yhat <= (myDict[1].yhat + myDict[2].yhat + myDict[3].yhat)*1.001))
        
if __name__ == '__main__':
    unittest.main()