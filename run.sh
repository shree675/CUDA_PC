# specific to me

rm -rf a.exe
nvcc $1 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" -rdc=true -lcudadevrt
./a
rm -rf *.exp
rm -rf *.lib