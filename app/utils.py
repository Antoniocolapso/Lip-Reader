########################################################################################################################
# ------------------------------------Template of AUTHOR: Omm AKA Antonio Colapso---------------------------------------#
########################################################################################################################
import time
import tensorflow as tf
from typing import List
import cv2
import os 
import bisect
import functools
import math
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import cmp_to_key, lru_cache, reduce
from heapq import heapify, heappop, heappush, heappushpop, nlargest, nsmallest
from io import BytesIO, IOBase
from itertools import accumulate, combinations, permutations
from operator import add, iand, ior, itemgetter, mul, xor
from string import ascii_lowercase, ascii_uppercase
from typing import *
import collections
import heapq
import itertools


# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')


def vsInput():
    sys.stdin = open('inputf.in', 'r')
    sys.stdout = open('outputf.out', 'w')


BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
def input(): return sys.stdin.readline().rstrip('\r\n')


#########################################################################################################
# ------------------------------------Functions By Antonio Colapso---------------------------------------#
# s


def SieveOfEratosthenes(n):
    prime = [True for i in range(n+1)]
    p = 2
    l = []
    while (p * p <= n):
        if (prime[p] == True):
            for i in range(p * p, n+1, p):
                prime[i] = False
        p += 1
    for p in range(2, n+1):
        if prime[p]:
            l.append(p)
            # print(p)
        else:
            continue
    return l


def isPrime(n):
    prime_flag = 0

    if n > 1:
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                prime_flag = 1
                break
        if prime_flag == 0:
            return True
        else:
            return False
    else:
        return False


def gcdofarray(a):
    x = 0
    for p in a:
        x = math.gcd(x, p)
    return x


def printDivisors(n):
    i = 1
    ans = []
    while i <= math.sqrt(n):
        if (n % i == 0):
            if (n / i == i):
                ans.append(i)
            else:
                ans.append(i)
                ans.append(n // i)
        i = i + 1
    ans.sort()
    return ans


def binaryToDecimal(n):
    return int(n, 2)


def countTriplets(a, n):
    s = set()
    for i in range(n):
        s.add(a[i])
    count = 0
    for i in range(n):
        for j in range(i + 1, n, 1):
            xr = a[i] ^ a[j]
            if xr in s and xr != a[i] and xr != a[j]:
                count += 1
    return int(count // 3)


def countOdd(L, R):
    N = (R - L) // 2
    if (R % 2 != 0 or L % 2 != 0):
        N += 1
    return N


def isPalindrome(s):
    return s == s[::-1]


def sufsum(a):
    test_list = a.copy()
    test_list.reverse()
    n = len(test_list)
    for i in range(1, n):
        test_list[i] = test_list[i] + test_list[i - 1]
    return test_list


def prsum(b):
    a = b.copy()
    for i in range(1, len(a)):
        a[i] += a[i - 1]
    return a


def badachotabadachota(nums):
    nums.sort()
    i = 0
    j = len(nums) - 1
    ans = []
    cc = 0
    while len(ans) != len(nums):
        if cc % 2 == 0:
            ans.append(nums[j])
            j -= 1
        else:
            ans.append(nums[i])
            i += 1
        cc += 1
    return ans


def chotabadachotabada(nums):
    nums.sort()
    i = 0
    j = len(nums) - 1
    ans = []
    cc = 0
    while len(ans) != len(nums):
        if cc % 2 == 1:
            ans.append(nums[j])
            j -= 1
        else:
            ans.append(nums[i])
            i += 1
        cc += 1
    return ans


def primeFactors(n):
    ans = []
    while n % 2 == 0:
        ans.append(2)
        n = n // 2

    for i in range(3, int(math.sqrt(n))+1, 2):
        while n % i == 0:
            ans.append(i)
            n = n // i
    if n > 2:
        ans.append(n)
    return ans


def closestMultiple(n, x):
    if x > n:
        return x
    z = (int)(x / 2)
    n = n + z
    n = n - (n % x)
    return n


def getPairsCount(arr, n, sum):
    m = [0] * 1000
    for i in range(0, n):
        m[arr[i]] += 1
    twice_count = 0
    for i in range(0, n):
        twice_count += m[int(sum - arr[i])]
        if (int(sum - arr[i]) == arr[i]):
            twice_count -= 1
    return int(twice_count / 2)


def remove_consec_duplicates(test_list):
    res = [i[0] for i in itertools.groupby(test_list)]
    return res


def BigPower(a, b, mod):
    if b == 0:
        return 1
    ans = BigPower(a, b//2, mod)
    ans *= ans
    ans %= mod
    if b % 2:
        ans *= a
    return ans % mod


def nextPowerOf2(n):
    count = 0
    if (n and not (n & (n - 1))):
        return n
    while (n != 0):
        n >>= 1
        count += 1
    return 1 << count


# This function multiplies x with the number
# represented by res[]. res_size is size of res[]
# or number of digits in the number represented
# by res[]. This function uses simple school
# mathematics for multiplication. This function
# may value of res_size and returns the new value
# of res_size
def multiply(x, res, res_size):

    carry = 0  # Initialize carry

    # One by one multiply n with individual
    # digits of res[]
    i = 0
    while i < res_size:
        prod = res[i] * x + carry
        res[i] = prod % 10  # Store last digit of
        # 'prod' in res[]
        # make sure floor division is used
        carry = prod//10  # Put rest in carry
        i = i + 1

    # Put carry in res and increase result size
    while (carry):
        res[res_size] = carry % 10
        # make sure floor division is used
        # to avoid floating value
        carry = carry // 10
        res_size = res_size + 1

    return res_size


def Kadane(a, size):

    max_so_far = -1e18 - 1
    max_ending_here = 0

    for i in range(0, size):
        max_ending_here = max_ending_here + a[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here

        if max_ending_here < 0:
            max_ending_here = 0
    return max_so_far


def highestPowerof2(n):
    res = 0
    for i in range(n, 0, -1):
        if ((i & (i - 1)) == 0):
            res = i
            break
    return res


def MEX(nList, start):
    nList = set(nList)
    mex = start
    while mex in nList:
        mex += 1
    return mex


def getClosest(val1, val2, target):

    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1


def tobinary(a): return bin(a).replace('0b', '')


def nextPerfectSquare(N):

    nextN = math.floor(math.sqrt(N)) + 1

    return nextN * nextN


def modFact(n, p):
    result = 1
    for i in range(1, n + 1):
        result *= i
        result %= p

    return result


def maxsubarrayproduct(arr):

    n = len(arr)
    max_ending_here = 1
    min_ending_here = 1
    max_so_far = 0
    flag = 0

    for i in range(0, n):

        if arr[i] > 0:
            max_ending_here = max_ending_here * arr[i]
            min_ending_here = min(min_ending_here * arr[i], 1)
            flag = 1

        elif arr[i] == 0:
            max_ending_here = 1
            min_ending_here = 1

        else:
            temp = max_ending_here
            max_ending_here = max(min_ending_here * arr[i], 1)
            min_ending_here = temp * arr[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here

    if flag == 0 and max_so_far == 0:
        return 0
    return max_so_far


def ncr(n, r, p):
    num = den = 1
    for i in range(r):
        num = (num * (n - i)) % p
        den = (den * (i + 1)) % p
    return (num * pow(den, p - 2, p)) % p


def sqrtl(x):
    if (x == 0 or x == 1):
        return x
    start = 1
    end = x//2
    while (start <= end):
        mid = (start + end) // 2
        if (mid*mid == x):
            return mid
        if (mid * mid < x):
            start = mid + 1
            ans = mid
        else:
            end = mid-1

    return ans


def ceill(a, b):
    return (a // b) + int(a % b != 0)


def LcmOfArray(arr, idx):

    # lcm(a,b) = (a*b/gcd(a,b))
    if (idx == len(arr)-1):
        return arr[idx]
    a = arr[idx]
    b = LcmOfArray(arr, idx+1)
    return int(a*b / math.gcd(a, b))


def base_change_Excluding_certain_Digits_or_Base_conversion(number_of_hated_digits, list_of_hated_digits, find_nth_number):
    digits1 = '0123456789'
    digits = ""  # only digits we want
    for i in digits1:
        if int(i) not in list_of_hated_digits:
            digits += i
    # final integer base after excluding "t" hated digits
    base = 10-number_of_hated_digits
    sol = []
    while find_nth_number:
        sol.append(digits[find_nth_number % base])
        find_nth_number //= base
    sol.reverse()
    return ("".join(sol))
# print(base_change_Excluding_certain_Digits_or_Base_conversion(2,[1,8],10))


def counter(l):
    cts = {}
    for x in l:
        cts[x] = cts.get(x, 0) + 1
    return cts


def mimic_string(sm,sn):   #//minimum steps to make both string equal
    prev_dp = list(range(len(sm) + 1))
    curr_dp = [0] * (len(sm) + 1)

    for i in range(1, len(sn) + 1):
        curr_dp[0] = i
        for j in range(1, len(sm) + 1):
            if sn[i - 1] == sm[j - 1]:
                curr_dp[j] = prev_dp[j - 1]
            else:
                curr_dp[j] = min(prev_dp[j], prev_dp[j - 1], curr_dp[j - 1]) + 1
        prev_dp = curr_dp[:]    
    return(curr_dp[len(sm)])

def LCS_length(s, t):
    m, n = len(s), len(t)
    prev_dp = [0] * (n + 1)
    curr_dp = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr_dp[j] = prev_dp[j - 1] + 1
            else:
                curr_dp[j] = max(prev_dp[j], curr_dp[j - 1])
        prev_dp = curr_dp[:]
    
    return curr_dp[n]

    lcs = ""
    i, j = m, n
    while i > 0 and j > 0:
        if s[i - 1] == t[j - 1]:
            lcs = s[i - 1] + lcs
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs

def kadane(arr):
    n = len(arr)
    max_ending_here = arr[0]
    max_so_far = arr[0]

    for i in range(1, n):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far


# #######################################################################################################################################
class segtree():
  def __init__(self,init,func,ide):
    self.n=len(init)
    self.func=func
    self.ide=ide
    self.size=1<<(self.n-1).bit_length()
    self.tree=[self.ide for i in range(2*self.size)]
    for i in range(self.n):
      self.tree[self.size+i]=init[i]
    for i in range(self.size-1,0,-1):
      self.tree[i]=self.func(self.tree[2*i], self.tree[2*i|1])
  
  def update(self,k,x):
    k+=self.size
    self.tree[k]=x
    k>>=1
    while k:
      self.tree[k]=self.func(self.tree[2*k],self.tree[k*2|1])
      k>>=1
  
  def get(self,i):
    return self.tree[i+self.size]
  
  def query(self,l,r):
    l+=self.size
    r+=self.size
    l_res=self.ide
    r_res=self.ide
    while l<r:
      if l&1:
        l_res=self.func(l_res,self.tree[l])
        l+=1
      if r&1:
        r-=1
        r_res=self.func(self.tree[r],r_res)
      l>>=1
      r>>=1
    return self.func(l_res,r_res)

  def max_right(self,l,f):
    assert 0<=l<=self.n 
    assert f(self.ide)
    if l==self.n:
      return self.n
    l+=self.size
    res=self.ide
    while True:
      while l&1==0:
        l>>=1
      if not f(self.func(res,self.tree[l])):
        while l<self.size:
          l*=2
          if f(self.func(res,self.tree[l])):
            res=self.func(res,self.tree[l])
            l+=1
        return l-self.size
      res=self.func(res,self.tree[l])
      l+=1
      if l&(-l)==l:
        break
    return self.n
  
  def min_left(self,r,f):
    assert 0<=r<=self.n
    assert f(self.ide)
    if r==0:
      return 0
    r+=self.size
    res=self.ide
    while True:
      r-=1
      while r>1 and r&1:
        r>>=1
      if not f(self.func(self.tree[r],res)):
        while r<self.size:
          r=2*r+1     
          if f(self.func(self.tree[r],res)):
            res=self.func(self.tree[r],res)
            r-=1
        return r+1-self.size
      res=self.func(self.tree[r],res)
      if r&(-r)==r:
        break
    return 0
    
  def __getitem__(self,i):
    return self.get(i)
  
  def __setitem__(self,key,val):
    self.update(key,val)

  def __iter__(self):
    for i in range(self.n):
      yield self.tree[i+self.size]
          
  def __str__(self):
    return str(list(self))



# #######################################################################################################################################


alphabets = list("abcdefghijklmnopqrstuvwxyz")
vowels = list("aeiou")
MOD1 = int(1e9 + 7)
MOD2 = 998244353
INF = int(1e17)


######################################################################################
# --------------------------------------Input-----------------------------------------#
######################################################################################
def I():
    return input()


def II():
    return int(input())


def MII():
    return map(int, input().split())


def LI():
    return list(input().split())


def LII():
    return list(map(int, input().split()))


def GMI():
    return map(lambda x: int(x) - 1, input().split())


def LGMI():
    return list(map(lambda x: int(x) - 1, input().split()))


inf = float('inf')

# ########################################################################
# #-----------------------------Code Here--------------------------------#
##########################################################################
#



vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    #print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = os.path.splitext(os.path.basename(path))[0]
    
    # Define the base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 's1'))
    alignment_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'alignments', 's1'))

    # Construct the full paths
    video_path = os.path.join(data_dir, f'{file_name}.mpg')
    alignment_path = os.path.join(alignment_dir, f'{file_name}.align')

    # Check if the files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"Alignment file {alignment_path} does not exist.")
    
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

# def load_data(path: str): 
#     path = bytes.decode(path.numpy())
#     file_name = path.split('/')[-1].split('.')[0]
#     # File name splitting for windows
#     file_name = path.split('\\')[-1].split('.')[0]
#     video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
#     alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
#     frames = load_video(video_path) 
#     alignments = load_alignments(alignment_path)
    
#     return frames, alignments
