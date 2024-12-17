import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeRegressor, plot_tree

class Solution:
	def canPlaceFlowers(self, flowerbed, n) -> bool:
		count = 0
		i = 0
		length = len(flowerbed)		
		while i < length:
			if flowerbed[i] == 0:
				emp_left = (i== 0 or flowerbed[i-1] == 0)
				emp_right = (i == length -1 or flowerbed[i+1] == 0)
				print(emp_left, emp_right, i, length)
				if emp_left and emp_right:
					flowerbed[i] =1
					count+=1
					i+=1
			i+=1
		if count >= n:
			return True
		else:
			return False
	def reverseVowels(self, s: str) -> str:
		vo = ['a','e','i','o','u', 'A', 'E', 'I', 'O', 'U']
		original_str = list(s)
		new_char= []
		new_index= []
		for i in range(0, len(original_str)):
			if original_str[i] in vo:
				new_char.append(original_str[i])
				new_index.append(i)
		print(new_char, new_index)		
		new_char.reverse()
		for j , k in zip(new_char, new_index):
			original_str[k] = j
		return "".join(original_str)
	def reverseWords(self, s: str) -> str:
		split_str = s.split()
		return split_str[::-1]	 
	def dictonary_learnings(self):
		
		self.d1 = dict([('audi','1'),('benz', '2'), ('Tata','3'), ('maru', '4')])
		

	def pandas_learnings(self, dataset):
		self.dictonary_learnings()
		val = pd.DataFrame(dataset)
		a = [1,2,3,4,5,6,7]
		new = pd.Series(a)
		new1 = pd.Series(a, index=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
		new2 = self.d1
		new3 = pd.Series(new2)
		new4 = pd.Series(new2, index = ["audi", "Tata"]) #index is for columns 
		#new10 = pd.DataFrame(new2, columns= ["audi", "Tata"])
		#print(new10)
		print(new4)
		print(new1['G'])
		print("NEW2",new2)
		print(new3)
		print("NEW4", new4)
		dataframe_with_2_series= {"A": [1,2,3,4,5], "B": [6,7,8,9,0]}
		new5 = pd.DataFrame(dataframe_with_2_series)
		new6= pd.DataFrame(dataframe_with_2_series, index= ['D1','D2','D3','D4','D5'])
		print(new5)
		print(new6.loc[['D1', 'D4']])
		new7 = pd.read_csv('dummy.csv')
		print("New7 ",new7.to_string())
		print("Head: ",new7.head())
		print(new7.tail(10))
		print(new7.info())
		print(pd.options.display.max_rows)

		new8 = pd.read_csv('dummy2.csv')
		print(new8.info())
		print(new8.isna().sum())
		#new8.fillna(130, inplace=True)
		x = new8['time'].median()
		new8['time'].fillna(x, inplace=True)
		print(new8.drop_duplicates(inplace=True))
		print(new8)
		#print(new8.dropna(subset= ["time", "card"]))
		#print(new8.dropna(subset= ["time", "card"]).info())
		#print(new8.dropna(axis=1, thresh=2))






		#new8.dropna(inplace=True)
	#	#print(new8.info)
		#print(new8.shape)
		new9 = new8.query('time > 1500 & card < 2000')
		print(new9.to_string())
		return val

#class Solution:
	def findMaxAverage(self, List, k) -> float:
		new_list = []
		for i in range(0, k):
			for j in range(len(List)):
				print(List[j])
		return List
	def learning_numpy(self):
		list1 = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
		print(list1)
		print(list1.ndim)
		print(list1[1][1])
		print(list1[0:3, 1:4])
		list2 = np.array([1,2,3,4,5,6,7,8,9,10])
		print("List2:",list2[-3:-1])
		#print(np.array_split(list2,2))
		print(np.where(list2 == 4))
		list3 = random.randint(100, size=(3,11))
		print(list3)
		print(list3.ndim, list3[0:3,1])

		list4 = random.normal(loc = 1, scale= 1, size=(3,5))
		#sns.displot(random.normal(size=1000))
		#plt.show()
		#list5 = np.array([[1,2,3,4],[5,6,7,8],[0,9,8,7]],[[1,2,3,4],[5,6,7,8],[0,9,8,7]])
		#print("list5: ", list5[0:3, 1], list5.ndim)
		list6 = np.array([1,2,3,4,5,6,7,8,9,0])
		print("list5: ", np.sqrt(list6), np.exp(list6), np.sin(list6))
		list7 = np.array([1,2,3,4,5,6])
		list8 = list7.view()
		list9 = list7.copy()
		list7[0]=41
		print("List7, list8:", list7, list8, list9)
		print("list9: ", list9.reshape(2,3))
		list10 = np.array([[[[1,2,3,4,5],[6,7,8,9,0]],[[1,2,3,4,5],[6,7,8,9,0]]], [[[1,2,3,4,5],[6,7,8,9,0]],[[1,2,3,4,5],[6,7,8,9,0]]]])
		print("list10: ", list10.ndim)
		for x in np.nditer(list10):
			print(x)

		list11 = np.array([11,2,43,6,523,32])
		print(np.where(list11%2 == 0))
		filtered = list11%2 == 0
		print("List11 : ", list11[filtered])
		list12 = np.array([1,2,3,4])
		list13 = np.array([1,2,3,4])
		print(np.sum([list12, list13], axis=0))
		print(np.sum([list12, list13], axis=1))
		return 0
	def learning_decisiontree(self):
		dataset = np.array( 
		[['Asset Flip', 100, 1000], 
		['Text Based', 500, 3000], 
		['Visual Novel', 1500, 5000], 
		['2D Pixel Art', 3500, 8000], 
		['2D Vector Art', 5000, 6500], 
		['Strategy', 6000, 7000], 
		['First Person Shooter', 8000, 15000], 
		['Simulator', 9500, 20000], 
		['Racing', 12000, 21000], 
		['RPG', 14000, 25000], 
		['Sandbox', 15500, 27000], 
		['Open-World', 16500, 30000], 
		['MMOFPS', 25000, 52000], 
		['MMORPG', 30000, 80000] 
		]) 
		print(dataset)
		X_dataset = dataset[:, 1].astype(int).reshape(-1, 1)
		Y_dataset = dataset[:, 2].astype(int).reshape(-1, 1)
		X_train, X_test, y_train, y_test = train_test_split(X_dataset, Y_dataset, test_size=0.2, random_state=42)
		print("X_train:",X_train)
		print("##############")
		print("Y_train:", y_train)
		print("X_test:",X_test)
		print("##############")
		print("Y_test:", y_test)
		regressor = DecisionTreeRegressor(random_state=0, max_depth=4)
		regressor.fit(X_train, y_train)
		y_pred = regressor.predict(X_test).astype(int)
		

		#print(X_dataset.shape, Y_dataset.shape)
		#regressor = DecisionTreeRegressor(random_state = 0)
		#regressor.fit(X_dataset, Y_dataset) 
		#y_pred = regressor.predict([[3750]]).astype(int)
		#print(len(y_test), len(y_pred))
		mse = mean_squared_error(y_test, y_pred)
		mae = mean_absolute_error(y_test, y_pred)
		rmse = np.sqrt(mse)
		r2 = r2_score(y_test, y_pred)
		print("Mean Squared Error:", mse)
		print("Mean Absolute Error:", mae)
		print("R2_score:", r2)
		
		x_pred = regressor.predict([[3750]]).astype(int)
		#print("Prediction from decision tree: ", x_pred)
		#scores = cross_val_score(regressor, X_dataset, Y_dataset, cv=5, scoring='r2')
		#print("Cross-Validation R² Scores:", scores)
		#X_grid = np.arange(X_dataset.min(), X_dataset.max(), 0.01) 
		#X_grid = X_grid.reshape(-1, 1)
		#plt.scatter(X_dataset, Y_dataset, color = 'red') 
		#print(X_grid)
		#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')  
		#plt.title('Profit to Production Cost (Decision Tree Regression)')  
		#plt.xlabel('Production Cost') 
		#plt.ylabel('Profit') 
		#
		#plt.show()
		plt.figure(figsize=(12, 8))
		plot_tree(regressor, feature_names=['Production Cost'], filled=True, rounded=True, fontsize=10)
		plt.title('Decision Tree Visualization')
		#plt.show()
		return x_pred
	def learning_RF(self):
		df= pd.read_csv('profit.csv')
		print(df)
		X = df.iloc[:,1:2].values
		y = df.iloc[:,2].values
		label_encoder = LabelEncoder()
		x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
		x_numerical = df.select_dtypes(exclude=['object']).values
		x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
		print("X_train: ", X_train)
		print("y_train:  ", y_train)

		regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
		regressor.fit(X_train, y_train)
		try:
			oob_score = regressor.oob_score_
			print(f'Out-of-Bag Score: {oob_score}')
		except AttributeError:
			print('OOB score not available. Ensure oob_score=True is set.')
		y_pred = regressor.predict(X_test)
		mse = mean_squared_error(y_test, y_pred)
		print(f'Mean Squared Error: {mse}')
		r2 = r2_score(y_test, y_pred)
		print(f'R-squared: {r2}')	
		x_pred = regressor.predict([[3750]])
		return x_pred
def main():
	solution = Solution()
	#flowerbed = [1,0,0,0,1]
	#n =1
	#s = "IceCreAm"
	#canplant = solution.canPlaceFlowers(flowerbed, n)
	#vow = solution.reverseVowels(s)
	#print(canplant)
	#s = "Hi hellow how"
	#new_strng_rev = " ".join(solution.reverseWords(s))
	#print(new_strng_rev)
	#data_set = {"cars":["Volvo", "Audi", "Benz"], "Type":["A", "B", "C"]}
	#dict_dataset = {1: "Geek", 2:"For", 3:"Learning"}
	#result=solution.pandas_learnings(data_set)
	#result2 = solution.dictonary_learnings()
	#List = [10,2,3,6,1,5]
	#result3 = solution.findMaxAverage(List, k =4)
	#print(result)
	solution.learning_numpy()
	#val1 = solution.learning_decisiontree()
	#val2 = solution.learning_RF()
	#print("The final predications are: ", val1, val2)
if __name__ == "__main__":
	main()
	

# Dictonary : A = {a : "1", b: "2"}
#				A= dict{[(a,'1'), (b,'2')]}

#JSON:

#B= {"a": {'0':10, '1':20},"b": {'0':20, '1':30}} 