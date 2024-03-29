from PIL import Image,ImageDraw
import colorsys
import os
import struct



LAYER_WIDTH=50
NODE_WIDTH=20
NODE_DISTANCE=20
EDGE_WIDTH=5

ACTIVATION_FUNCTION_TANH=0
ACTIVATION_FUNCTION_STEP=1
ACTIVATION_FUNCTION_LINEAR=2
ACTIVATION_FUNCTION_RELU=3
_ACTIVATION_FUNCTION_FLAG_INPUT=4



def _generate_image(name,input_count,output_count,node_count,edge_count,nodes,edges):
	bias_range=max(max(map(lambda e:abs(e[0]),nodes.values())),0.0001)
	weight_range=max(max(map(abs,edges.values())),0.0001)
	node_layer_position={}
	layer_element_count=[[]]
	for i in nodes.keys():
		if (i<input_count):
			node_layer_position[i]=(0,i)
			layer_element_count[0].append(i)
			continue
		if (i>=node_count-output_count):
			continue
		max_layer=0
		for j in nodes.keys():
			if (edges[(i,j)]==0.0):
				continue
			if (j not in node_layer_position):
				edges[(i,j)]=0.0
				continue
			max_layer=max(max_layer,node_layer_position[j][0])
		max_layer+=1
		if (max_layer==len(layer_element_count)):
			layer_element_count.append([i])
		else:
			layer_element_count[max_layer].append(i)
		node_layer_position[i]=(max_layer,len(layer_element_count[max_layer])-1)
	width,height=len(layer_element_count),max(map(len,layer_element_count))
	layer_element_count.append([])
	width+=1
	for i in range(node_count-output_count,node_count):
		layer_element_count[-1].append(i)
		node_layer_position[i]=(width-1,i-node_count+output_count)
	for i,(x,y) in node_layer_position.items():
		cx=x*(LAYER_WIDTH+NODE_WIDTH)+NODE_WIDTH/2
		cy=y*(NODE_DISTANCE+NODE_WIDTH)+NODE_WIDTH/2+(height-len(layer_element_count[x]))*(NODE_DISTANCE+NODE_WIDTH)/2
		node_layer_position[i]=(cx,cy)
	image=Image.new("RGBA",(width*(LAYER_WIDTH+NODE_WIDTH)-LAYER_WIDTH,height*(NODE_DISTANCE+NODE_WIDTH)-NODE_DISTANCE),(0,0,0,0))
	draw=ImageDraw.Draw(image)
	for i in nodes.keys():
		for j in nodes.keys():
			weight=edges[(i,j)]
			if (weight==0.0):
				continue
			t=int(255*min(max(weight/(2*weight_range)+0.5,0),1))
			draw.line((node_layer_position[i],node_layer_position[j]),fill=(t,t,t),width=EDGE_WIDTH)
	for i,(cx,cy) in node_layer_position.items():
		color=(colorsys.hsv_to_rgb(min(max(nodes[i][0]/(2*bias_range)+0.5,0),1)/3,1,1) if nodes[i][2] else (0.85,0.15,0.85))
		type=nodes[i][1]
		if (type&_ACTIVATION_FUNCTION_FLAG_INPUT):
			type&=~_ACTIVATION_FUNCTION_FLAG_INPUT
			color=(0.15,0.65,0.65)
		if (type==ACTIVATION_FUNCTION_TANH):
			draw.ellipse((cx-NODE_WIDTH/2,cy-NODE_WIDTH/2,cx+NODE_WIDTH/2,cy+NODE_WIDTH/2),fill=tuple(map(lambda x:int(255*x),color)),width=0)
		elif (type==ACTIVATION_FUNCTION_STEP):
			draw.rectangle((cx-NODE_WIDTH/2,cy-NODE_WIDTH/2,cx+NODE_WIDTH/2,cy+NODE_WIDTH/2),fill=tuple(map(lambda x:int(255*x),color)),width=0)
		elif (type==ACTIVATION_FUNCTION_LINEAR):
			draw.rectangle((cx-NODE_WIDTH/2,cy-NODE_WIDTH/2,cx+NODE_WIDTH/2,cy+NODE_WIDTH/2),fill=None,width=5,outline=tuple(map(lambda x:int(255*x),color)))
		else:
			draw.regular_polygon((cx,cy,NODE_WIDTH),3,fill=tuple(map(lambda x:int(255*x),color)),outline=None)
	image.save(f"build/{name}.png")



for name in os.listdir("build"):
	if (not name.endswith(".neat")):
		continue
	with open(f"build/{name}","rb") as rf:
		input_count,output_count,node_count,edge_count=struct.unpack("<IIII",rf.read(16))
		nodes={i:(0.0,ACTIVATION_FUNCTION_LINEAR|_ACTIVATION_FUNCTION_FLAG_INPUT,True) for i in range(0,input_count)}
		edges={(i//node_count,i%node_count):0.0 for i in range(0,node_count*node_count)}
		for i in range(input_count,node_count):
			node=struct.unpack("<fB?",rf.read(6))
			if (i>=node_count-input_count-output_count):
				node=(node[0],node[1],True)
			nodes[i]=node
		for i in range(0,edge_count):
			j,weight=struct.unpack("<If",rf.read(8))
			edges[(j//node_count,j%node_count)]=weight
		_generate_image(name[:-5],input_count,output_count,node_count,edge_count,nodes,edges)
		for i in range(input_count,node_count-output_count):
			if (not nodes[i][2]):
				del nodes[i]
		_generate_image(name[:-5]+".reduced",input_count,output_count,node_count,edge_count,nodes,edges)

