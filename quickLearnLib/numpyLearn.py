#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def numpyTest(  ):
    # basics
    a = np.zeros( (5,2) ) 
    print( 'ndim : ', a.ndim, 'shape : ', a.shape, 'size : ', a.size, 'dtype : ', a.dtype )

    # array creation
    # array, zeros, zeros_like, ones, ones_like, empty, empty_like,
    # arange, linspace, numpy.random.rand, numpy.random.randn, fromfunction, fromfile
    a = np.array( [1,2,3,4,5] )
    print( 'a:', a, 'dtype:', a.dtype )
    b = np.array( [1.2, 2.2, 4.3] )
    print( 'b:', b, 'dtype:', b.dtype )
    c = np.array( [1,2,3,4], dtype=complex )
    print( 'c:', c, 'dtype:', c.dtype )
    a = np.zeros( (2,3) )
    b = np.ones( (2,3) )
    print( 'a:', a, '\n', '\rb:', b )
    a = np.arange( 10,30,5 )
    b = np.linspace( 10,30,200 )
    print( 'b:', b, 'size:', b.size, 'shape:', b.shape )
    a = np.random.rand( 3,2 )
    b = np.random.randint( 0,1,20 )
    print( a,b )
    print( np.ones( 5 ).ndim )
    print( np.max( [1,2,3,4,5] ), np.argmax( [1,10,3,4,5] ), np.argsort( [1,2,3,7,3,45] ) )
    print( a, np.max( a, axis=0 ) )

    # shape manipulation
    a = np.random.rand( 2,5 ) * 10+5
    print( a, a.ravel(), a.T, a.reshape(5,2) )
    print( a )
    print( np.transpose( a ) )
    print( a )
    b = np.ones( (2,2) )
    print( np.hstack( (a,b) ) )

def pandasTest(  ):
    # object creation
    s = pd.Series( np.int32( np.random.rand( 10 )*10 ) )
    dates = pd.date_range( '20130101',periods=6 )
    df = pd.DataFrame( np.random.randn( 6,4 ),index=dates, columns=list('ABCD') )
    print( df )
    df2 = pd.DataFrame( {'A':1,
        'B':pd.Timestamp('20130102'),
        'C':pd.Series(1,index=list(range(4)),dtype='float64' ),
        'D':np.array([3]*4,dtype='int32'),
        'E':pd.Categorical(['test','train','test','train']),
        'F':'foo' }
        )
    print( df2, df2.dtypes )

    print( df.head(10) )
    print( df.tail(3) )
    print( df.index )
    print( df.columns )
    print( df.values )

    print( df.describe() )
    print( df.T )
    print( df )

    print( df.sort_index( axis=1,ascending=False ) )
    print( df.sort_values( by='B' ) )
    print( df['A'] )

    s1 = pd.Series( np.arange(1,7),index=pd.date_range('20130102', periods=6) )
    df['F'] = s1
    df.at[dates[0],'A'] = 0
    df.iloc[0,1] = 0
    print( df )

    df1 = df.reindex( index=dates[0:4], columns=list(df.columns)  )
    print( df1 )
    print( df1.dropna( how='any' ) )
    print( df1.fillna(value=5) )
    print( pd.isna(df1) )
    
    print( df )
    print( df['A'] )

    df = pd.DataFrame( np.random.rand(5,6), columns=list('ABCDEF') )
    print( df )
    s = df.iloc[3]
    print(s)
    k = df.append( s, ignore_index=True )
    print( k )

if __name__ == '__main__' :



