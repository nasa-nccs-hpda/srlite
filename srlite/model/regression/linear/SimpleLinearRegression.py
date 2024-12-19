import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression(object):

	def __init__(self, x, y):
		self = self
		self.x = x
		self.y = y

	def estimate_coef_orig(self, x, y):
		# number of observations/points
		n = np.size(x)

		# mean of x and y vector
		m_x = np.mean(x)
		m_y = np.mean(y)

		# calculating cross-deviation and deviation about x
		SS_xy = np.sum(y * x) - n * m_y * m_x
		SS_xx = np.sum(x * x) - n * m_x * m_x

		# calculating regression coefficients
		b_1 = SS_xy / SS_xx
		b_0 = m_y - b_1 * m_x

		return (b_0, b_1)

	def estimate_coef(self, _x, _y):
		# number of observations/points
		n = np.size(_x)

#		nanval = x[0,0]
#		print(nanval)
		# mean of x and y vector
#		m_x = np.mean(x)
#		m_y = np.mean(y)

		# mean of x and y vector - replace nan with -9999
		x = np.nan_to_num(_x, copy=True, nan = -9999)
		y = np.nan_to_num(_y, copy=True, nan = -9999)
#		x = np.where(x != nanval, x, -9999)
		# mean of x and y vector - replace 0 with nan and ignore all nans
		#		x = np.where(x != 0, x, np.nan)
#		y = np.where(y != 0, y, np.nan)
#		m_x = np.nanmean(np.where(x != 0, x, np.nan), 1)#
#		m_y = np.nanmean(np.where(y != 0, y, np.nan), 1)

#		xma = np.ma.masked_values(x.ReadAsArray(), -9999)
		xma = np.ma.masked_values(x, -9999)
		yma = np.ma.masked_values(y, -9999)
		x = xma
		y = yma

		# mean of x and y vector - ignore all nans
#		m_x = np.mean(x)
#		m_y = np.mean(y)
		m_x = np.nanmean(x)
		m_y = np.nanmean(y)

		# calculating cross-deviation and deviation about x
#		SS_xy = np.sum(y*x) - n*m_y*m_x
#		SS_xx = np.sum(x*x) - n*m_x*m_x

		# calculating cross-deviation and deviation about x - ignore nan
		SS_xy = np.nansum(y*x) - n*m_y*m_x
		SS_xx = np.nansum(x*x) - n*m_x*m_x

		# calculating regression coefficients
		b_1 = SS_xy / SS_xx
		b_0 = m_y - b_1*m_x

		return (b_0, b_1)

	def plot_regression_line(self, x, y, b):
		# plotting the actual points as scatter plot
		plt.scatter(x, y, color = "m",
				marker = "o", s = 30)

		# predicted response vector
		y_pred = b[0] + b[1]*x

		# plotting the regression line
		plt.plot(x, y_pred, color = "g")

		# putting labels
		plt.xlabel('x')
		plt.ylabel('y')

		# function to show plot
		plt.show()

	def run(self):

		# estimating coefficients
		b = self.estimate_coef(self.x, self.y)
		print("Estimated coefficients:\nb_0 = {} \
			\nb_1 = {}".format(b[0], b[1]))
#		print("Estimated coefficients:\n y-int? (b_0) = {} \
#			\n slope? (b_1) = {}".format(b[0], b[1]))

		# plotting regression line
#		self.plot_regression_line(self.x, self.y, b)
		return b

if __name__ == "__main__":

	# observations / data
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

	lr = SimpleLinearRegression(x,y)
	lr.run()

