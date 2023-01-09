from PIL import Image,ImageDraw
import colorsys
import itertools
import os
import struct



LAYER_WIDTH=50
NODE_WIDTH=20
NODE_DISTANCE=20



for name in os.listdir("../build"):
	if (not name.endswith(".neat")):
		continue
	with open(f"../build/{name}","rb") as rf:
		input_count,output_count,node_count,edge_count=struct.unpack("<IIII",rf.read(16))
		nodes=[0.0 for _ in range(0,input_count)]
		edges=[0.0 for _ in range(0,node_count*node_count)]
		for i in range(0,node_count-input_count):
			nodes.append(struct.unpack("<f",rf.read(4))[0])
		for i in range(0,edge_count):
			j,weight=struct.unpack("<If",rf.read(8))
			edges[j]=weight
		bias_range=max(map(abs,nodes))
		weight_range=max(map(abs,edges))
		node_layer_position=[None for _ in range(0,node_count)]
		layer_element_count=[[]]
		for i in range(0,input_count):
			node_layer_position[i]=(0,i)
			layer_element_count[0].append(i)
		for i in itertools.chain(range(input_count+output_count,node_count),range(input_count,input_count+output_count)):
			max_layer=0
			for j in range(0,node_count):
				if (edges[i*node_count+j]==0.0):
					continue
				if (node_layer_position[j] is None):
					edges[i*node_count+j]=0.0
					continue
				max_layer=max(max_layer,node_layer_position[j][0])
			max_layer+=1
			if (max_layer==len(layer_element_count)):
				layer_element_count.append([i])
			else:
				layer_element_count[max_layer].append(i)
			node_layer_position[i]=(max_layer,len(layer_element_count[max_layer])-1)
		for i,(x,y) in enumerate(node_layer_position):
			cx=x*(LAYER_WIDTH+NODE_WIDTH)+NODE_WIDTH/2
			cy=y*(NODE_DISTANCE+NODE_WIDTH)+NODE_WIDTH/2
			node_layer_position[i]=(cx,cy)
		image=Image.new("RGBA",(len(layer_element_count)*(LAYER_WIDTH+NODE_WIDTH)-LAYER_WIDTH,max(map(len,layer_element_count))*(NODE_DISTANCE+NODE_WIDTH)-NODE_DISTANCE),(0,0,0,0))
		draw=ImageDraw.Draw(image)
		for i in range(0,node_count):
			for j in range(0,node_count):
				weight=edges[i*node_count+j]
				if (weight==0.0):
					continue
				t=int(255*min(max(weight/(2*weight_range)+0.5,0),1))
				draw.line((node_layer_position[i],node_layer_position[j]),fill=(t,t,t),width=5)
		for i,(cx,cy) in enumerate(node_layer_position):
			t=min(max(nodes[i]/(2*bias_range)+0.5,0),1)
			draw.rectangle((cx-NODE_WIDTH/2,cy-NODE_WIDTH/2,cx+NODE_WIDTH/2,cy+NODE_WIDTH/2),fill=tuple(map(lambda x:int(255*x),colorsys.hsv_to_rgb(t/3,1,1))),width=0)
		image.save(f"../build/{name[:-5]}.png")
