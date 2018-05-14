

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


###############################################################
#Function Name:my_random_exmples
#Function input:number of the class
#Function output100 examples
#Function Action:creates 100 examples of class which distributes normally with mean : 2 * tag and
#standart deviation (sigma): 1.
################################################################
def my_random_exmples(m_tag):
    #creates new 100 samples of givven tag, With a normal distribution
    return np.random.normal(m_tag*2,1.0,100)


###############################################################
#Function Name: my_softmax_function
#Function input: m_xt - example,m_w - classes features matrix
#                       m_b- constants vector, tag - the class
#Function output:probability that xt belongs to class i + 1
#                according to parameters w and b
#Function Action: the function calcuslate the result of softmax function
################################################################
def my_softmax_function(m_b,m_w,m_xt,tag):
    # dominator = 0
    # for j in range(3):
    #     dominator += np.exp(m_w[j] * m_xt + m_b[j])
    # return np.exp(m_w[tag] * m_xt + m_b[tag]) / dominator
    m_sum=0
    for i in range(3):
        m_sum = m_sum+np.exp(m_b[i]+m_w[i]*m_xt)
    numerator = np.exp(m_b[tag]+ m_xt*m_w[tag])
    return numerator/m_sum

###############################################################
#Function Name:
#Function input:
#Function output
#Function Action:
################################################################
def create_set(set):
    # for tag in range(1, 4): # creating examples for
    #     examples = my_random_exmples(tag)
    #     for example in examples:
    #         set.append((example, tag))
    #
    for i in range(1,4):
        my_examples = my_random_exmples(i)
        for x in my_examples:
            set.append((x,i))



###############################################################
#Function Name:training_Action
#Function input:w(classes features matrix),b(constants vector)
#       eta(training rate),and  training set
#Function output:none
#Function Action:training set in order to find the optimal w and b.
#using logistic regression method using SGD update rule, according to
#teoritichal part
################################################################
def training_Action(set,m_eta,b,w):
    epochs = 30
    for k in range(epochs):
        np.random.shuffle(set)
        for (x,tag) in set:
            #update rule of w and b:
            for i in range(3):
                if(tag != i+1):
                    wTag = x*my_softmax_function(b,w,x,i)
                    bTag = my_softmax_function(b,w,x,i)
                else:
                    wTag = x * my_softmax_function(b, w, x, i) -x
                    bTag = my_softmax_function(b, w, x, i)-1

                #update rule:
                b[i] = b[i] - bTag*m_eta
                w[i] = w[i] -wTag*m_eta

#Calculation of the value of the foams function, according to the formula in the case of a normal distribution
###############################################################
#Function Name:Density_function_of_normal_distribution
#Function input: mean,standard deviation,x
#Function output:output of normal distribution density function
#Function Action:Calculation of the value of the foams function,
#  according to the formula in the case of a normal distribution
###############################################################


def Density_function_of_normal_distribution(x,sigma,expectation):
    return (1.0 / (math.sqrt(2*math.pi) * sigma))*np.exp((-(x-expectation)**2) /(2*(sigma**2)))

###############################################################
#Function Name:Conditional_Probability
#Function input:x,y
#Function output:output of normal distribution density function accoding to exercize request
#Function Action:call Density_function_of_normal_distribution function with right arguments
################################################################
def Conditional_Probability(x,y):
    return Density_function_of_normal_distribution(x,1,2*y)

###############################################################
#Function Name:plot_Result
#Function input:w: classes features matrix,b: constants vector
#Function output: none
#Function Action:plotting graph of the real distribution and the softmax distribution according to w, b
# in range [0, 10]
################################################################
def plot_Result(b,w):
    training_output={}
    for i in range(0,11):
        training_output[i] = my_softmax_function(b,w,i,0)
    real_output={}
    for i in range(0, 11):
        real_output[i] = (Conditional_Probability(i,1)/(Conditional_Probability(i,1) + Conditional_Probability(i,2)+
                                                        Conditional_Probability(i,3)))

    redLine, =plt.plot(real_output.keys(),real_output.values(),"red",label="Real")
    BlueLine, = plt.plot(training_output.keys(), training_output.values(), "blue", label="Softmax Distribution")
    plt.legend(handler_map={redLine:HandlerLine2D(numpoints=4)})
    plt.show()



def main():
    #first initialize parameteres b and w
    b = [0,0,0]
    w = [0,0,0]

    #define our learning rate:
    m_eta = 0.1
    #define our training set
    m_set = []
    create_set(m_set)
    training_Action(m_set,m_eta,b,w)
    #training(w, b, m_eta, m_set)
    plot_Result(b,w)

if __name__ == "__main__":
    main()

