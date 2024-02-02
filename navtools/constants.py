'''
|========================================= constants.py ===========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/constants.py                                                                 |
|  @brief    Common navigational constants.                                                        |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Tanner Koza <kozatanner@gmail.com>                                                    |
|            Daniel Sturdivant <sturdivant20@gmail.com>                                            |
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

# Physical
SPEED_OF_LIGHT = 299792458                  #! [m/s]
BOLTZMANN = 1.38e-23                        #! [J/K]
GRAVITY = 9.80665                           #! Earth's gravity constant [m/s^2]

# Conversions
G2T = 1e-4                                  #! Gauss to Tesla
FT2M = 0.3048                               #! Feet to meters

# Earth Models
WGS84_R0 = 6378137.0                        #! WGS84 Equatorial radius (semi-major axis) [m]
WGS84_RP = 6356752.31425                    #! WGS84 Polar radius (semi-major axis)
WGS84_E = 0.0818191908425                   #! WGS84 eccentricity
WGS84_E2 = WGS84_E*WGS84_E                  #! WGS84 eccentricity squared
WGS84_MU = 3.986004418e14                   #! WGS84 earth gravitational constant
WGS84_F = (WGS84_R0 - WGS84_RP) / WGS84_R0  #! WGS84 flattening
WGS84_J2 = 1.082627E-3;                     #! WGS84 earth second gravitational constant

# GNSS
TOTAL_SATS = 132                            #! total number of GNSS satellites
GNSS_PI = 3.1415926535898                   #! pi constant defined for GNSS
GNSS_TWO_PI = 2.0 * GNSS_PI                 #! 2*pi
GNSS_HALF_PI = 0.5 * GNSS_PI                #! pi/2
GNSS_R2D = 180.0 / GNSS_PI                  #! radians to degrees
GNSS_D2R = GNSS_PI / 180.0                  #! degrees to radians
GNSS_OMEGA_EARTH = 7.2921151467e-5          #! Earth's rotational constant
