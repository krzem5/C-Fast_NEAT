import subprocess
import sys
import os



DEFAULT_ARGS=[]



if (not os.path.exists("build")):
	os.mkdir("build")
if ("--release" in sys.argv):
	fl=[]
	for r,_,cfl in os.walk("src"):
		r=r.replace("\\","/").strip("/")+"/"
		for f in cfl:
			if (f[-2:]==".c"):
				fl.append(f"build/{(r+f).replace('/','$')}.o")
				if (subprocess.run(["gcc","-Wall","-lm","-Werror","-march=native","-mno-avx256-split-unaligned-load","-ffast-math","-momit-leaf-frame-pointer","-Ofast","-c",r+f,"-o",f"build/{(r+f).replace('/','$')}.o","-Isrc/include","-g"]).returncode!=0):
					sys.exit(1)
	if (subprocess.run(["gcc","-o","build/fast_neat"]+fl+["-lm"]).returncode!=0):
		sys.exit(1)
else:
	fl=[]
	for r,_,cfl in os.walk("src"):
		r=r.replace("\\","/").strip("/")+"/"
		for f in cfl:
			if (f[-2:]==".c"):
				fl.append(f"build/{(r+f).replace('/','$')}.o")
				if (subprocess.run(["gcc","-Wall","-g","-lm","-Werror","-march=native","-O0","-c",r+f,"-o",f"build/{(r+f).replace('/','$')}.o","-Isrc/include"]).returncode!=0):
					sys.exit(1)
	if (subprocess.run(["gcc","-g","-o","build/fast_neat"]+fl+["-lm"]).returncode!=0):
		sys.exit(1)
if ("--run" in sys.argv):
	subprocess.run(["build/fast_neat"]+DEFAULT_ARGS)
