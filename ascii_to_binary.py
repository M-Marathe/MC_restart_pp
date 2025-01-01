# Python program to illustrate the
# conversion of ASCII to Binary

# Importing binascii module
import binascii

# Initializing a ASCII string
#Text = "ABC123"
Text_ascii = input()

# Calling the a2b_uu() function to
# Convert the ascii string to binary
Binary = binascii.a2b_uu(Text_ascii)
# Getting the Binary value
print(type(Binary))
print("Binary for the ", Text_ascii, " = ", Binary)

Text_binary = bytes(input())
# Calling the b2a_uu() function to
# Convert the binary string to ascii
Ascii = binascii.b2a_uu(Text_binary)
print("ASCII for the ", Text_binary, " = ", Ascii)
