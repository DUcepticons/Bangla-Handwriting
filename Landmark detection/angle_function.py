# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:05:17 2021

@author: Raiya
"""

import math
PI = 3.14159265

#get angle between two bodypart lines
def angle(line_a, line_b):
    a_x0, a_y0, a_x1, a_y1 = line_a
    b_x0, b_y0, b_x1, b_y1 = line_b
    
    '''
    #exceptions if either line is parallel to y axis
    if (a_x1 - a_x0) == 0 and (b_x1 - b_x0)==0:
        theta = 0
    
    elif (a_x1 - a_x0) == 0:
        theta = PI/2 - math.atan2((b_y1 - b_y0), (b_x1 - b_x0))

    elif (b_x1 - b_x0) == 0:
        theta = PI/2 - math.atan2((a_y1 - a_y0), (a_x1 - a_x0))
        
    #normal case
    else:
    '''
        
    m_a = (a_y1 - a_y0)/(a_x1 - a_x0)
    m_b = (b_y1 - b_y0)/(b_x1 - b_x0)
    
    if m_a*m_b != -1:
        tan_theta = (m_a-m_b)/(1+m_a*m_b)
        tan_theta = tan_theta if tan_theta>=0 else -tan_theta #always positive
        theta_1 = math.atan(tan_theta)
        theta_2 = PI - theta_1
    elif m_a == m_b: 
        theta_1 = 0
        theta_1 = PI - 0
    else:
        theta_1, theta_2 = PI/2, PI/2
        
            
            
    
    theta_1_deg = math.degrees(theta_1)
    theta_2_deg = math.degrees(theta_2)
    
    
    return min(theta_1_deg, theta_2_deg)

line_a = [331,127,195,203]
line_b = [195,203,338,315]


print(angle(line_a,line_b))