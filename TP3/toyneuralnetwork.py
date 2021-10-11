#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 01:33:44 2020

@author: benoitmerlet
"""

import numpy as np
from numpy import random as nprd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from copy import deepcopy as dcp

############################################################################
"""                      The class ToyPb                                """
############################################################################

class ToyPb:  
    """ 
    class of simple classification problems (2D sets) 
      - f: a numerical functions of 2 variables (a length 2 numpy array)
      - bounds : list of 4 floats
      - loss, loss_prime: two numerical functions of 1 variable 
          (represent the cost function of the error and its derivative)
      - coef_reg : float number
    """
    def __init__(self, **kwargs):
        name = kwargs.get('name', None)
        self.name = name
        "name, f, bounds, dim, loss, loss_prime, coef_reg"
        ######  criteria function  ###################
        dim = kwargs.get("dim",None) or 2
        self.dim =dim 
        
        self.coef_reg = kwargs.get("coef_reg")
        
        f =kwargs.get('f', None)
        if f:
            self.f = f
        elif name == "sin": 
            
            self.f = lambda X : X[1] - np.sin(np.pi*X[0])
        elif name == "affine":
            self.f =lambda X : 2*X[0] - X[1] - .5
        elif name == "disc":
            self.f = lambda X : X[0]**2 + X[1]**2 - .25
        elif name == "square":
            self.f = lambda X : max(np.abs(X[0]), np.abs(X[1])) - .5
        elif name == "ring":
            self.f = lambda X : 9/64 - (13/16)*(X[0]**2 + X[1]**2) + (X[0]**2 + X[1]**2)**2
        elif name == "ncube":
            self.f = lambda X : max(np.abs(x) for x in X) - np.sqrt(.5)*(.5)**(1/dim)
        elif name == "nball":
            self.f = lambda X : sum(x**2 for x in X) - (.25)**(2/dim)

        
        bounds= kwargs.get('bounds', None) or (-1,1)
        bounds = (2*dim//len(bounds))*bounds
        self.bounds = bounds 
        
        ######  LOSS function and its derivative ###################
        loss = kwargs.get('loss', None)
        if loss == "softplus":
            self.loss = lambda s :np.log(1 + np.exp(-s))
            self.loss_prime = lambda s : -1/(1 + np.exp(s))
            self.loss_seconde = lambda s : np.exp(s)/(1 + np.exp(s))**2
        elif loss == "demanding":
            EPS = 0.1
            self.loss = lambda s : np.sqrt(EPS + (s - 1)**2) - s + 1
            self.loss_prime = lambda s : (s - 1)/np.sqrt(EPS + (s - 1)**2) - 1 
            self.loss_seconde = lambda s :EPS/(np.sqrt((s - 1)**2 + EPS)**3)
        elif loss:
            self.loss = loss
            self.loss_prime = kwargs.get('loss_prime', None)
        else:
            EPS = 0.1
            self.loss = lambda s : np.sqrt(EPS + s**2) - s
            self.loss_prime = lambda s : s/np.sqrt(EPS + s**2) - 1
            self.loss_seconde =lambda s : EPS/(np.sqrt(s**2 + EPS)**3)

        
    def criteria(self, **kwargs):
        dim = kwargs.get("dim",None) or 2
        
        bounds= kwargs.get('bounds', None) or (-1,1)
        bounds =  (2*dim//len(bounds))*bounds
        self.bounds = bounds 
        
        f =kwargs.get('f', None)
        if f:
            self.f = f
        else:
            name= kwargs.get('name', None)
            if name == "sin": 
                self.f = lambda X : X[1] - np.sin(np.pi*X[0])
            elif name == "affine":
                self.f =lambda X : 2*X[0] - X[1] - .5
            elif name == "disc":
                self.f = lambda X : X[0]**2 + X[1]**2 - .25
            elif name == "square":
                self.f = lambda X : np.maximum(np.abs(X[0]), np.abs(X[1])) - .5
            elif name == "ring":
                self.f = lambda X : 9/64 - (13/16)*(X[0]**2 + X[1]**2) + (X[0]**2 + X[1]**2)**2
            elif name == "ncube":
                self.f = lambda X : max(np.abs(x) for x in X) - .5
            elif name == "nball":
                self.f = lambda X : sum(x**2 for x in X) - .25
  
    def define_loss(self, **kwargs):
        loss = kwargs.get('f', None)
        if loss == "softplus":
            self.loss = lambda s :np.log(1 + np.exp(-s))
            self.loss_prime = lambda s : -1/(1 + np.exp(s))
            self.loss_seconde = lambda s : np.exp(s)/(1 + np.exp(s))**2
        elif loss == "demanding":
            EPS = 0.1
            self.loss = lambda s : np.sqrt(EPS + (s - 1)**2) - s + 1
            self.loss_prime = lambda s : (s - 1)/np.sqrt(EPS + (s - 1)**2) - 1 
            self.loss_seconde = lambda s :EPS/(np.sqrt((s - 1)**2 + EPS)**3)
        elif loss:
            self.loss = loss
            self.loss_prime = kwargs.get('loss_prime', None)
        else:
            EPS = 0.1
            self.loss = lambda s : np.sqrt(EPS + s**2) - s
            self.loss_prime = lambda s : s/np.sqrt(EPS + s**2) - 1
            self.loss_seconde = lambda s :EPS/(np.sqrt(s**2 + EPS)**3)

            
    def show_border(self, style='r-'):
        name = self.name
        if name == "sin": 
            g = lambda x : np.sin(np.pi*x)
            xx = np.linspace(-1, 1, 100)
            plt.plot(xx, g(xx), style)
        elif name == "affine":
            g =lambda x : 2*x - .5
            xx = np.linspace(-1,1,3)
            plt.plot(xx,g(xx),style)
        elif name == "disc":
            theta=np.linspace(0,2*np.pi,100)
            plt.plot(.5*np.cos(theta),.5*np.sin(theta),style)
        elif name == "square":
            a, b = -.5,.5
            plt.plot([a, a, b, b, a], [a, b, b, a, a], style)
        elif name == "ring":
            theta=np.linspace(0,2*np.pi,100)
            plt.plot(.5*np.cos(theta),.5*np.sin(theta),style)
            plt.plot( .75*np.cos(theta), .75*np.sin(theta),style)          
        else:
            plt.plot([0],[0],'.') 
        plt.axis('equal')
        #plt.axis(self.bounds)
        
############################################################################
"""                     The class nD_data                               """
############################################################################
class nD_data:
    """
    dim, n:  two non-negative integers
    X : 2D numpy array of shape n x dim 
    Y : numpy array of length n 
    Ypred : numpy array of length n 
    """
    
    def __init__(self, **kwargs):
        """  ('X','Y') or ('n', 'dim', 'bounds', ('f' or 'pb') ), [ 'init_pred' ] """
        
        X = kwargs.get("X", None)
        self.Y = None
        pb = kwargs.get("pb", None)
        try:  
            self.n, self.dim = X.shape
            self.X = X
            self.Y = kwargs.get("Y", None)
        except:    
            n = kwargs.get("n", None)
            if n:
                self.n, dim = n, kwargs.get("dim", None) or (pb and pb.dim) or 2
                bounds = kwargs.get("bounds", None) or (pb and pb.bounds) or (-1,1)
                bounds= bounds*(2*dim//len(bounds))
                X = nprd.rand(n, dim)
                for k in range(dim):
                    X[:, k] = bounds[2*k] + (bounds[2*k + 1] - bounds[2*k])*X[:, k]
                self.X, self.bounds, self.dim  = X, bounds, dim
        f=kwargs.get("f")
        if self.n and (f or pb) and self.Y == None :
            f = f or pb.f                     
            n, X, Y  = self.n, self.X, np.zeros(n)
            for i in range(n):
                Y[i] = 2*(f(X[i]) > 0) - 1
            self.Y = Y            
            
        if self.n and kwargs.get("init_pred"):
            self.Ypred = np.zeros(self.n)
        
    def prediction(self, **kwargs):
        """ NN [zero_one] """
#       self.Ypred = self.Ypred or np.zeros(self.n)
        Ypred=self.Ypred
        NN = kwargs.get("NN")
        if kwargs.get("zero_one"):
            for j in range(self.n):
                Ypred[j] = 2*(NN.output(self.X[j]) > 0) - 1  
        else: 
            for j in range(self.n):
                Ypred[j] = NN.output(self.X[j])
    
    def init_pred(self):
        self.Ypred = np.zeros(self.n)
            
    def classify(self, pb):
        X, n = self.X, self.n 
        Y = np.zeros([n,1])
        for i in range(n):
            Y[i] = 2*(pb.f(X[i]) > 0) - 1
        self.Y = Y
    
    def show_class(self, pred = None, wrong_only=None):
        if pred:
            X, Y = self.X, self.Ypred
        else:
            X, Y = self.X, self.Y
        if wrong_only:
            ind = np.where(self.Y*Y < 0)
            plt.scatter(X[ind,0], X[ind,1], c='k', s=3, label="misclass.")
        else:
            ind = np.where(Y>0)
            plt.scatter(X[ind,0], X[ind,1], c='b', s=3, label=r"$y=1$")
            ind  = np.where(Y<0)
            plt.scatter(X[ind,0], X[ind,1], c='r', s=3, label=r"$y=-1$")
        plt.axis(self.bounds)      


############################################################################
"""                     The class ToyNN                                  """
############################################################################
class ToyNN:
    """ 
    class of simple Neural Networks 
      - N : an integer. N - 1 is the number of hidden layers of the NN, so N is at least 1 
      - card : a list of int which represent the number of nodes in each layer, 
            Its length is at least 2. It starts with a 2 and ends with a 1, [2, a1, a2, ..., 1]
      - W : list of N 2D numpy arrays, W[n][i,j] is the weight of the edge between 
            the iest node of layer n and the jiest node of layer n + 1.  
      - Bias : list of N 1D numpy arrays, Bias[n][i] is the bias on node i of layer n + 1   
      - Nparam : integer. The number of parameters in the NN
      - chi : numerical function of 1 variable (the activation function)
      - chi_prime : numerical function of 1 variable (derivative of the former) 
      - xx, yy, zz : three 2D arrays for the graphic representation of the NN's output 
    """

    def __init__(self, **kwargs):
        """ [ 'card', 'coef_bounds', 'chi="tanh', 'chi_prime', 'grid', 'shadow_nn', 'no_vector_init' ] """
        self.test=kwargs
        card = kwargs.get("card", None)
        self.card = card
        N = len(card) - 1
        self.N = N
        Nparam = 0          # Nwb is the number of parameters to be optimized (nbr of entries of W + nbr of entries of Bias) 
        for n in range(N):
            Nparam += (card[n] + 1)*card[n + 1]
        self.Nparam = Nparam
            
        chi = kwargs.get("chi", None)    
        if chi == "tanh":
            self.chi = lambda x : np.tanh(x)
            self.chi_prime = lambda x : 1/np.cosh(x)**2
        elif chi == "sigmoide":
            self.chi = lambda x : 1/(1 + np.exp(-x))
            self.chi_prime = lambda x : np.exp(x)/(1 + np.exp(x))**2
        elif chi == "RELU":
            self.chi = lambda s : np.where(s > 1, s - 1, 0)
            self.chi_prime = lambda s : np.where(s > 1, 1, 0)
        elif chi:
            self.chi=chi
            self.chi_prime =  kwargs.get("chi_prime") 
            
        W  = kwargs.get("W", None)
        Bias =  kwargs.get("Bias", None)
        
        coef_bounds = kwargs.get("coef_bounds", None)
        self.coef_bounds = coef_bounds
        if coef_bounds:
            a, b, c, d = coef_bounds
            W, Bias = [], []
            for n in range(N):
                W.append((b - a)*nprd.rand(card[n],card[n + 1]) + a)
                Bias.append((d - c)*nprd.rand(card[n + 1]) + c)
            self.W, self.Bias = W, Bias
        elif coef_bounds==0:
            W, Bias = [], []
            for n in range(N):
                W.append(np.zeros([card[n],card[n + 1]]))
                Bias.append(nprd.rand(card[n + 1])) 
            self.W, self.Bias = W, Bias

        if kwargs.get('shadow_nn'):
            W, Bias = [], []
            for n in range(N):
                W.append(np.zeros([card[n],card[n + 1]]))
                Bias.append(nprd.rand(card[n + 1])) 
            self.W2, self.Bias2 = W, Bias
            
        if not kwargs.get("no_vector_init", None):   
            self.DW, self.DBias =[],[]
            for n in range(N):
                self.DW.append(np.zeros([card[n], card[n + 1]]))
                self.DBias.append(np.zeros(card[n + 1]))

        grid = kwargs.get("grid", None)  
        if grid:
            if len(grid) == 3:
                a, b, c, d, nx, ny = grid[0], grid[1],grid[0], grid[1], grid[2], grid[2]
            elif len(grid) == 6:
                a, b, c, d, nx, ny = grid[0], grid[1], grid[2], grid[3], grid[4], grid[5]               
            x=np.linspace(a, b ,nx)
            y=np.linspace(c, d, ny)
            self.xx, self.yy = np.meshgrid(x,y)
            self.zz = np.zeros(self.xx.shape)
            
    def add_to_coefs(self, deltaW, deltaBias):
        for n in range(self.N):
            self.W[n] += deltaW[n]
            self.Bias[n] += deltaBias[n]
            
    def init_vector(self, **kwargs):
        N = self.N
        DW = kwargs.get('DW') or self.DW
        DBias = kwargs.get('DBias') or self.DBias
        for n in range(N):
            DW[n] = np.zeros([self.card[n], self.card[n + 1]])
            DBias[n] = np.zeros(self.card[n + 1])
            
    def add_to_vector(self, dW, dBias, **kwargs):
        DW = kwargs.get("DW") or self.DW
        DBias = kwargs.get("DBias") or self.DBias
        c = kwargs.get("c")
        if c:
            for n in range(self.N):
                DW[n] += c*dW[n]
                DBias[n] += c*dBias[n]
        else:
            for n in range(self.N):
                DW[n] += dW[n]
                DBias[n] += dBias[n]
            
    def add_vectors(self, V_W, V_Bias, d_W, d_Bias, output = False, c=None):
        if output:
            out_W, out_Bias = [], []
            if c:
                for n in range(self.N):
                    out_W.append(V_W[n] + c*d_W[n])
                    out_Bias.append(V_Bias[n] + c*d_Bias[n])  
            else:
                for n in range(self.N):
                    out_W.append(V_W[n] + d_W[n])
                    out_Bias.append(V_Bias[n] + d_Bias[n])  
            return[out_W, out_Bias]
        else:
            if c:
                for n in range(self.N):
                    V_W[n] += c*d_W[n]
                    V_Bias[n] += c*d_Bias[n]
            else:
                for n in range(self.N):
                    V_W[n] += d_W[n]
                    V_Bias[n] += d_Bias[n]
                    
    
    def dot(self, dW, dBias, **kwargs):
        DW = kwargs.get("DW") or self.DW
        DBias = kwargs.get("DBias") or self.DBias
        s= 0
        for n in range(self.N):
            s += np.sum(DW[n] * dW[n])
            s += np.sum(DBias[n] * dBias[n])
        return s
    
    def add_vector_to_coefs(self,**kwargs):
        """["c", "DBias", "DW" ] """
        DW = kwargs.get("DW") or self.DW
        DBias = kwargs.get("DBias") or self.DBias
        c =kwargs.get("c", None)
        if c != None:
            for n in range(self.N):
                self.W[n] += c*DW[n]
                self.Bias[n] += c*DBias[n]
        else:
            for n in range(self.N):
                self.W[n] += DW[n]
                self.Bias[n] += DBias[n]
    
    def add_vector_to_coefs_in_shadow(self,**kwargs):
        """["c", "DBias", "DW" ] """
        DW = kwargs.get("DW") or self.DW
        DBias = kwargs.get("DBias") or self.DBias
        c =kwargs.get("c", None)
        if c != None:
            for n in range(self.N):
                self.W2[n] = self.W[n] + c*DW[n]
                self.Bias2[n] = self.Bias[n] + c*DBias[n]
        else:
            for n in range(self.N):
                self.W2[n] = self.W[n] + DW[n]
                self.Bias2[n] = self.Bias[n] + DW[n]
                
    def init_vector_to_one(self, **kwargs):
        DW = kwargs.get("DW") or self.DW
        DBias = kwargs.get("DBias") or self.DBias
        c =kwargs.get("c")
        if c:
            for n in range(self.N):
                DW[n] = c*np.ones([self.card[n], self.card[n + 1]])
                DBias[n] = c*np.ones(self.card[n + 1])
        else:
            for n in range(self.N):
                DW[n] = np.ones([self.card[n], self.card[n + 1]])
                DBias[n] = np.ones(self.card[n + 1])          
                  
    def create_zero_vector(self):
        dW, dBias = [], []
        for n in range(self.N):
            dW.append(np.zeros([self.card[n],self.card[n + 1]]))
            dBias.append(np.zeros(self.card[n + 1])) 
        return [dW, dBias]
                
    def mult_vector(self, c, **kwargs):
        V_W = kwargs.get("V_W") or self.DW
        V_Bias = kwargs.get("V_Bias") or self.DBias
        if kwargs.get("output"):
            out_W, out_Bias = [], []
            for n in range(self.N):
                out_W.append(c*V_W[n])
                out_Bias.append(c*V_Bias[n])
            return [out_W, out_Bias]
        else:
            for n in range(self.N):
                V_W[n] *= c
                V_Bias[n] *= c
            
    def V_VTranspose_dot(self,V_W, V_Bias, d_W, d_Bias, c=None, output=False):
        c = self.dot(V_W, V_Bias, DW=d_W, DBias=d_Bias)*(c or 1)
        if output:
            return self.mult_vector(c, V_W= V_W, V_Bias= V_Bias, output = True)
        else:
            d_W, d_Bias = self.mult_vector(c, V_W= V_W, V_Bias= V_Bias, output = True)
            
    def copy_vector(self, dW, dBias):
        copy_dW, copy_dBias = [], []
        for n in range(self.N):
            copy_dW.append(dW[n].copy())
            copy_dBias.append(dW[n].copy())
        return [copy_dW, copy_dBias]
    # Forward computation of the output value for a data X
    def output(self, X):
        o = np.array(X)
        for n in range(self.N -1):
            i = self.W[n].T.dot(o) + self.Bias[n]
            o = self.chi(i)
        return  (self.W[-1].T.dot(o) + self.Bias[-1])[0]
    
    def output_shadow(self, X):
        o = np.array(X)
        for n in range(self.N -1):
            i = self.W2[n].T.dot(o) + self.Bias2[n]
            o = self.chi(i)
        return  (self.W2[-1].T.dot(o) + self.Bias2[-1])[0]

    # Gradient of output with respect to W and Bias (multplied by c)
    def descent(self, **kwargs):
        """ ['X', 'y',, 'f', 'tau', 'pb', 'add_to_vector', 'new', 'add_to_coefs'] """
        # Forward computation of input and output values at the nodes 
        o = np.array(kwargs.get("X"))
        O, I =[], [o]    
        for n in range(self.N):
            O.append(o)      
            i = self.W[n].T.dot(o) + self.Bias[n]
            I.append(i)
            o = self.chi(i)

        # Backward computation of the gradients
        tau, y, pb, N  = kwargs.get("tau"), kwargs.get("y"), kwargs.get("pb"), self.N
        f = kwargs.get("f") or (pb and pb.loss_prime) or 1  
        if f==1:
            desc_bias = np.array([1.])
        else:
            desc_bias = -tau*y*f(y*i)
        if kwargs.get("new"):
            for n in range(N - 1, 0, -1):
                self.DBias[n] = desc_bias
                self.DW[n] = np.tensordot(O[n], desc_bias, 0)
                desc_bias = self.chi_prime(I[n])*(self.W[n].dot(desc_bias))
            self.DBias[0] = desc_bias
            self.DW[0] = np.tensordot(O[0], desc_bias, 0)              
        elif kwargs.get("add_to_vector"):
            for n in range(N - 1, 0, -1):
                self.DBias[n] += desc_bias
                self.DW[n] += np.tensordot(O[n], desc_bias, 0)
                desc_bias = self.chi_prime(I[n])*(self.W[n].dot(desc_bias))
            self.DBias[0] += desc_bias
            self.DW[0] += np.tensordot(O[0], desc_bias, 0) 
        elif kwargs.get("add_to_coefs"):
            for n in range(N - 1, 0, -1):
                self.Bias[n] += desc_bias
                self.W[n] += np.tensordot(O[n], desc_bias, 0)
                desc_bias = self.chi_prime(I[n])*(self.W[n].dot(desc_bias))
            self.Bias[0] += desc_bias
            self.W[0] += np.tensordot(O[0], desc_bias, 0) 
            return i 
        else:
            Desc_W, Desc_Bias = [], []            
            for n in range(N - 1, 0, -1):
                Desc_Bias.append(desc_bias)
                Desc_W.append(np.tensordot(O[n], desc_bias, 0))
                desc_bias = self.chi_prime(I[n])*(self.W[n].dot(desc_bias))
            Desc_Bias.append(desc_bias)
            Desc_W.append(np.tensordot(O[0], desc_bias, 0))
            Desc_W.reverse() ; Desc_Bias.reverse()
            if kwargs.get("output_nnoutput"): return [Desc_W, Desc_Bias, i]
            else: return [Desc_W, Desc_Bias]  

    def def_grid(self, grid):
        if len(grid) == 3:
            a, b, c, d, nx, ny = grid[0], grid[1],grid[0], grid[1], grid[2], grid[2]
        elif len(grid) == 6:
            a, b, c, d, nx, ny = grid[0], grid[1], grid[2], grid[3], grid[4], grid[5]               
        x=np.linspace(a,b,nx)
        y=np.linspace(c,d,ny)
        self.xx, self.yy = np.meshgrid(x,y)
        self.zz = np.zeros(self.xx.shape)
 
    def def_random_coefs(self, coef_bounds=None):
        if coef_bounds==0:
            W, Bias=[], []
            for n in range(self.N):
                W.append(np.zeros([self.card[n], self.card[n + 1]]))
                Bias.append(np.random.rand(self.card[n + 1])) 
        elif coef_bounds or self.coef_bounds:
            a, b, c, d = coef_bounds or self.coef_bounds
            W, Bias=[], []
            card = self.card
            for n in range(self.N):
                W.append((b - a)*nprd.rand(card[n],card[n + 1]) + a)
                Bias.append((d - c)*nprd.rand(card[n + 1]) + c)
        self.W, self.Bias = W, Bias
  
                     
    def  def_activation(self, chi = None, chi_prime= None):
        if chi == "sigmoide":
            self.chi = lambda x : 1/(1 + np.exp(-x))
            self.chi_prime = lambda x : np.exp(-x)/(1 + np.exp(-x))**2
        elif chi == "RELU":
            self.chi = lambda s : np.where(s > 1, s - 1, 0)
            self.chi_prime = lambda s : np.where(s > 1, 1, 0)            
        elif chi:
            self.chi=chi
            self.chi_prime =  chi_prime
        else:
            self.chi = lambda x : np.tanh(x)
            self.chi_prime = lambda x : 1/np.cosh(x)**2


    ### Computation of predictions (arrays of outputs) 
    def prediction(self, **kwargs):
        """ DATA  [zero_one] """
        DATA = kwargs.get("DATA")
        X, n = DATA.X, DATA.n     
        #DATA.Ypred = DATA.Ypred or np.zeros([n, self.card[0]])
        Ypred = DATA.Ypred
        if kwargs.get("zero_one"):
            for j in range(n):
                Ypred[j] = 2*(self.output(X[j]) > 0) - 1  
        else: 
            for j in range(n):
                Ypred[j] = self.output(X[j]) 
                
    #### total loss  
    def total_loss_shadow(self, **kwargs):
        """ DATA and { zero_one or pb or loss}"""
        DATA = kwargs.get("DATA")
        X, Y, n, cost = DATA.X, DATA.Y, DATA.n, 0
        if kwargs.get("zero_one"):
            for j in range(n):
                cost += Y[j]*self.output_shadow(X[j]) < 0
        else :
            loss = kwargs.get("loss") or kwargs.get("pb").loss
            for j in range(n): 
                cost += loss(Y[j]*self.output_shadow(X[j]))
        return cost/n
    
    def total_loss(self, **kwargs):
        """ DATA and { zero_one or pb or loss}"""
        DATA = kwargs.get("DATA")
        X, Y, n, cost = DATA.X, DATA.Y, DATA.n, 0
        if kwargs.get("zero_one"):
            for j in range(n):
                cost += Y[j]*self.output(X[j]) < 0
        else :
            loss = kwargs.get("loss") or kwargs.get("pb").loss
            for j in range(n): 
                cost += loss(Y[j]*self.output(X[j]))
        return cost/n
        
    ### total loss and prediction 
    def total_loss_and_prediction(self, **kwargs):
        """ DATA and { zero_one or pb or loss}"""
        DATA = kwargs.get("DATA")
        X, Y, n = DATA.X, DATA.Y, DATA.n     
        #DATA.Ypred = DATA.Ypred or np.zeros([n, self.card[0]])
        Ypred = DATA.Ypred
        cost = 0
        if kwargs.get("zero_one"):
            for j in range(n):
                ypred = 2*(self.output(X[j])>0) - 1
                cost += (1 - ypred*Y[j])//2
                Ypred[j] = ypred
        else: 
            loss = kwargs.get("loss") or kwargs.get("pb").loss
            for j in range(n):
                ypred = self.output(X[j]) 
                cost += loss(Y[j]*ypred)
                Ypred[j] = ypred
        return cost/n
    
    ### Displays the classification realized by self on the point of the grid
    def show_pred(self):
        nx,ny = self.xx.shape
        for i in range(nx):
            for j in range(ny):
                    self.zz[i,j] = self.output([self.xx[i,j], self.yy[i,j]])
        plt.imshow(self.zz,interpolation='bilinear', origin='lower',cmap=cm.RdYlBu, 
               vmin=-1, vmax=1,alpha =.6, extent=(-1, 1, -1, 1))
            
    ### Function for the graphical representation of the NN
    def show_benoit(self): 
        xright = 0
        nright = self.card[0]
        dl = .5/nright
        HR=np.linspace(dl,1-dl,nright)
        xright = 0
        for w in self.W:
            xleft = xright
            xright += np.sqrt(w.size)
            nleft, nright = nright, w.shape[1]
            dl = .5/nright
            HL, HR = HR, np.linspace(dl,1-dl,nright)
            MAX = np.amax(abs(w))
            for j in range(nleft):
                for k in range(nright):
                    val = w[j,k]
                    if val>0:
                        plt.plot([xleft, xright], [HL[j], HR[k]],color="darkorange",
                                 linestyle="dashed",linewidth=5*val/MAX)
                    else:
                        plt.plot([xleft, xright], [HL[j], HR[k]],color="plum",
                                 linestyle="dashed",linewidth=-5*val/MAX)
            plt.scatter([xleft]*nleft,HL, color = 'chocolate', s = 200)
        plt.scatter([xright]*nright,HR, color='chocolate', s = 200)
        plt.show()            
        
    def show(self):
        xright = 0
        nright = self.card[0]
        dl = .5/nright
        HR=np.linspace(dl,1-dl,nright)
        xright = 0
        for n in range(self.N):
            w = self.W[n]
            xleft = xright
            xright += np.sqrt(w.size)
            nleft, nright = nright, w.shape[1]
            dl = .5/nright
            HL, HR = HR, np.linspace(dl,1-dl,nright)
            MAX = np.amax(abs(w)) + 1e-12
            colors = ["tomato", "teal"]  # 0: negative, 1: positive
            for j in range(nleft):
                for k in range(nright):
                    val = w[j,k]
                    plt.plot([xleft, xright], [HL[j], HR[k]],color=colors[val>0],
                             lw=1.5, alpha=abs(val)/MAX, zorder=5)
            if n:
                b = self.Bias[n - 1]
                #MAX = np.amax(abs(b)) + 1e-12
                #bsize = abs(b)/MAX               
                bcolor = np.where(b<0,colors[0],np.where(b>0,colors[1], 'white'))
                #plt.scatter([xleft]*nleft,HL, 
                #        fc='dimgrey', s=10, lw=1.5, zorder=10)
                plt.scatter([xleft]*nleft,HL, ec='dimgrey', 
                        fc=bcolor, s=350, lw=1.5, zorder=10)
            else: plt.scatter([xleft]*nleft,HL, ec='dimgrey',
                              fc='white', s=350, lw=1.5, zorder=10)
        b=self.Bias[n]        
        bcolor= colors[0] if b<0 else colors[1] if b>0 else 'white'
        plt.scatter([xright]*nright,HR, color='dimgrey', 
                        fc=bcolor, s=350, lw=1.5, zorder=10)
        plt.axis('off')
        plt.show()  

############################################################################### 
"""                             EOF                                         """
###############################################################################