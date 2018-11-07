# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 00:07:47 2018

@author: Torronto
"""

import numpy as np
from itertools import permutations, combinations

def array2spacings(array):
    sorted_array = np.sort(array)
    spacings = np.zeros(len(array)-1)
    for i in range(0,len(array)-1):
        spacings[i] = sorted_array[i+1] - sorted_array[i]
    return spacings

def spacings2array(spacings):
    array = np.zeros(len(spacings)+1)
    for i in range(0,len(spacings)):
        array[i+1] = array[i] + spacings[i]
    return array

def array2coarray(array):
    ultimo = int(max(array))
    coarray = np.zeros(ultimo+1, dtype = np.int)
    for count1 in range(len(array)):
        for count2 in range(count1,len(array)):
            coarray[int(abs(array[count1]-array[count2]))] += 1
    return coarray

def spacings2coarray(spacings):
    ultimo = int(sum(spacings))
    coarray = np.zeros(ultimo+1, dtype = np.int)
    coarray[0] = 1
    for count1 in range(len(spacings)):
        for count2 in range(count1,len(spacings)+1):
            coarray[int(sum(spacings[count1:count2]))] += 1
    return coarray

def array_isnonredundant(array):
    ultimo = int(max(array))
    coarray = np.zeros(ultimo+1, dtype = np.int)
    for count1 in range(len(array)):
        for count2 in range(count1+1,len(array)):
            if coarray[int(abs(array[count1]-array[count2]))] == 1:
                return False
            else:
                coarray[int(abs(array[count1]-array[count2]))] += 1
    return True

def spacings_isnonredundant(spacings):
    ultimo = int(sum(spacings))
    coarray = np.zeros(ultimo+1, dtype = np.int)
    for count1 in range(len(spacings)):
        for count2 in range(count1+1,len(spacings)+1):
            if coarray[int(sum(spacings[count1:count2]))] == 1:
                return False
            else:
                coarray[int(sum(spacings[count1:count2]))] += 1
    return True
'''
def spacings_isnonredundant(spacings):
    ultimo = int(sum(spacings))
    coarray = np.zeros(ultimo+1, dtype = np.int)
    temp_soma = np.copy(spacings)
    coarray[temp_soma] += 1
    for i in range(1,len(spacings)):
        temp_soma[:len(spacings)-i] += spacings[i:len(spacings)]
        for j in temp_soma[:len(spacings)-i]: coarray[j] += 1
        if (coarray>1).any(): return False
    return True
'''
def get_nonredundant(n,extra_len = 0, nmax = 100):
    
    if n == 8:
        return [1, 3, 5, 6, 7, 10, 2]
    if n == 9:
        return [1,4,7,13,2,8,6,3]
    if n == 10:
        return [1,5,4,13,3,8,7,12,2]
    if n == 11:
        return [1,3,9,15,5,14,7,10,6,2]
    if n == 12:
        return [2,4,18,5,11,3,12,13,7,1,9]
    if n == 13:
        return [2,3,20,12,6,16,11,15,4,9,1,7]
    if n == 14:
        return [5,23,10,3,8,1,18,7,17,15,14,2,4]
    if n == 15:
        return [6,1,8,13,12,11,24,14,3,2,27,10,16,4]
    if n == 16:
        return [1,3,7,15,6,24,12,8,39,2,17,16,13,5,9]
    if n == 17:
        return [5,2,10,35,4,11,13,1,19,22,16,21,6,3,23,8]
    if n == 18:
        return [2,8,12,31,3,26,1,6,9,32,18,5,14,21,4,13,11]
    if n == 19:
        return [1,5,19,7,40,28,8,12,10,23,16,18,3,14,27,2,9,4]
    if n == 20:
        return [1,7,3,57,9,17,22,5,35,2,21,15,14,4,16,12,13,6,24]
    min_len = n*(n-1)/2
    nrs = []
    while 1:
        unordered_spacings = range(1,n+extra_len)
        for combination in combinations(unordered_spacings,n-1):
            if sum(combination) == min_len + extra_len:
                for spacings in permutations(combination):
                    if spacings_isnonredundant(spacings):
                        #return spacings
                        nrs.append(spacings)
                        if len(nrs)>=nmax:
                            if nmax == 1:
                                return nrs[0]
                            return nrs
        if len(nrs)>0:
            return nrs
        extra_len += 1

