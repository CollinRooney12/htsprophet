from setuptools import setup

setup(name = 'htsprophet',
      version = '0.0.1',
      description = "Creates Hierarchical Time Series Forecasts with Facebook's Prophet tool",
      url = "http://github.com/CollinRooney12/htsprophet",
      author = "Collin Rooney",
      author_email = 'CollinRooney12@gmail.com',
      license = 'MIT',
      keywords='hts time series hierarchy forecast Prophet',
      packages = ['htsprophet'],
      zip_safe = False,
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires = [
              'matplotlib',
              'pandas>=0.18.1',
              'numpy',
              'fbprophet',
              'sys',
              'scikit-learn>=0.18'],
       )