import subprocess
import sys
import os



DEFAULT_ARGS=[]



if (not os.path.exists("build")):
	os.mkdir("build")
if (os.name=="nt"):
	cd=os.getcwd()
	os.chdir("build")
	if ("--release" in sys.argv):
		if (subprocess.run(["cl","/Wv:18","/c","/permissive-","/Zc:preprocessor","/GS","/utf-8","/W3","/Zc:wchar_t","/Gm-","/sdl","/Zc:inline","/fp:precise","/D","NDEBUG","/D","_WINDOWS","/D","_UNICODE","/D","_CRT_SECURE_NO_WARNINGS","/D","UNICODE","/errorReport:none","/WX","/Zc:forScope","/Gd","/Oi","/FC","/EHsc","/nologo","/diagnostics:column","/GL","/Gy","/Zi","/O2","/Oi","/MD","/I","../src/include","../src/main.c","../src/fast_neat/*.c","../src/examples/*.c"]).returncode!=0 or subprocess.run(["link","*.obj","/OUT:fast_neat.exe","/DYNAMICBASE","/MACHINE:X64","/SUBSYSTEM:CONSOLE","/ERRORREPORT:none","/NOLOGO","/TLBID:1","/WX","/LTCG","/OPT:REF","/INCREMENTAL:NO","/OPT:ICF"]).returncode!=0):
			os.chdir(cd)
			sys.exit(1)
	else:
		if (subprocess.run(["cl","/Wv:18","/c","/permissive-","/Zc:preprocessor","/GS","/utf-8","/W3","/Zc:wchar_t","/Gm-","/sdl","/Zc:inline","/fp:precise","/D","_DEBUG","/D","_WINDOWS","/D","_UNICODE","/D","_CRT_SECURE_NO_WARNINGS","/D","UNICODE","/errorReport:none","/WX","/Zc:forScope","/Gd","/Oi","/FC","/EHsc","/nologo","/diagnostics:column","/ZI","/Od","/RTC1","/MDd","/I","../src/include","../src/main.c","../src/fast_neat/*.c","../src/examples/*.c"]).returncode!=0 or subprocess.run(["link","*.obj","/OUT:fast_neat.exe","/DYNAMICBASE","/MACHINE:X64","/SUBSYSTEM:CONSOLE","/ERRORREPORT:none","/NOLOGO","/TLBID:1","/WX","/DEBUG","/INCREMENTAL"]).returncode!=0):
			os.chdir(cd)
			sys.exit(1)
	os.chdir(cd)
	if ("--run" in sys.argv):
		subprocess.run(["build/fast_neat.exe"]+DEFAULT_ARGS)
else:
	if ("--release" in sys.argv):
		fl=[]
		for r,_,cfl in os.walk("src"):
			r=r.replace("\\","/").strip("/")+"/"
			for f in cfl:
				if (f[-2:]==".c"):
					fl.append(f"build/{(r+f).replace('/','$')}.o")
					if (subprocess.run(["gcc","-Wall","-lm","-Werror","-mavx","-mavx2","-mfma","-O3","-c",r+f,"-o",f"build/{(r+f).replace('/','$')}.o","-Isrc/include"]).returncode!=0):
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
					if (subprocess.run(["gcc","-Wall","-g","-lm","-Werror","-mavx","-mavx2","-mfma","-O0","-c",r+f,"-o",f"build/{(r+f).replace('/','$')}.o","-Isrc/include"]).returncode!=0):
						sys.exit(1)
		if (subprocess.run(["gcc","-g","-o","build/fast_neat"]+fl+["-lm"]).returncode!=0):
			sys.exit(1)
	if ("--run" in sys.argv):
		subprocess.run(["build/fast_neat"]+DEFAULT_ARGS)
