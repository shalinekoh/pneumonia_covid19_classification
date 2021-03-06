?	pw?n??@pw?n??@!pw?n??@	ƥ???X@ƥ???X@!ƥ???X@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6pw?n??@{????@1% &????@A?????.??I^????@Y?3???@*	?I?|pA2P
Iterator::Model::Prefetch???4^??@!?'???X@)???4^??@1?'???X@:Preprocessing2F
Iterator::Model˂??`??@!      Y@)I? OZ??1+?d??7(?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 96.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9ƥ???X@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{????@{????@!{????@      ??!       "	% &????@% &????@!% &????@*      ??!       2	?????.???????.??!?????.??:	^????@^????@!^????@B      ??!       J	?3???@?3???@!?3???@R      ??!       Z	?3???@?3???@!?3???@JGPUYƥ???X@b ?"8
functional_1/conv2d_2/Conv2DConv2D??F???!??F???"8
functional_1/conv2d_4/Conv2DConv2D?,?ߝ?!^??
????"-
IteratorGetNext/_1_Send?t-c???!Ҟ^?8??"9
functional_1/conv2d_72/Conv2DConv2D???%'??!v[?'???"8
functional_1/conv2d_1/Conv2DConv2D???9????!?,ۺ??"9
functional_1/conv2d_74/Conv2DConv2D?1??????!??Ke??"9
functional_1/conv2d_75/Conv2DConv2D\*?g%???!???/???">
"functional_1/block17_1_conv/Conv2DConv2Db(f?/̀?!????????"7
functional_1/conv_7b/Conv2DConv2Dx??_????!Jf??̴??"9
functional_1/conv2d_73/Conv2DConv2D???I}?!9'?????Q      Y@Y?@5???a?*??̯X@qцl?ª[?y3]+78?"?
host?Your program is HIGHLY input-bound because 96.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 