//
// Created by AnNingUI on 25-12-9.
//
#include "WCN/WCN_MATH_CPP.hpp"
#include <iostream>

using namespace WCN::Math;
constexpr Mat4 m1{
   0x00,0x01,0x02,0x03,
   0x10,0x11,0x12,0x13,
   0x20,0x21,0x22,0x23,
   0x30,0x31,0x32,0x33
};
constexpr Mat4 m2{
   0x00,0x01,0x02,0x03,
   0x10,0x11,0x12,0x13,
   0x20,0x21,0x22,0x23,
   0x30,0x31,0x32,0x33
};
int main()
{
    auto i = 0;
    std::cout << "m1:" << std::endl;
    for (const auto v : m1.m) 
    {
        std::cout << v;
        if (v < 10) 
        {
            std::cout << "   ";  // 3 spaces for single digit
        } 
        else if (v < 100) 
        {
            std::cout << "  ";   // 2 spaces for double digit
        } 
        else 
        {
            std::cout << " ";    // 1 space for triple digit
        }
        i++;
        if (i % 4 == 0) 
        {
            std::cout << std::endl;
        }
    }
    i = 0;
    std::cout << "m2:" << std::endl;
    for (const auto v : m2.m) 
    {
        std::cout << v;
        if (v < 10) 
        {
            std::cout << "   ";  // 3 spaces for single digit
        } 
        else if (v < 100) 
        {
            std::cout << "  ";   // 2 spaces for double digit
        } 
        else 
        {
            std::cout << " ";    // 1 space for triple digit
        }
        i++;
        if (i % 4 == 0) 
        {
            std::cout << std::endl;
        }
    }
    i = 0;
    std::cout << "m1 * m2 * 2 + m1:" << std::endl;
    for (const auto v : (m1 * m2 * 2 + m1).m) 
    {
        std::cout << v;
        if (v < 10)
        {
            std::cout << "      ";
        }
        else if (v < 100)
        {
            std::cout << "     ";
        }
        else if (v < 1000)
        {
            std::cout << "    ";
        }
        else if (v < 10000)
        {
            std::cout << "   ";
        }
        else
        {
            std::cout << "  ";
        }
        i++;
        if (i % 4 == 0) {
            std::cout << std::endl;
        }
    }
}