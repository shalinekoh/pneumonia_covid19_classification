?	Yک?41?@Yک?41?@!Yک?41?@	u?v?P@u?v?P@!u?v?P@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Yک?41?@-?\odq@1Q1??tC?@Ah%???¯?I?????@Y???)?j@*	IKƉ4A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??T????@!i??D?T@)??T????@1i??D?T@:Preprocessing2P
Iterator::Model::PrefetchI????8k@!????-0@)I????8k@1????-0@:Preprocessing2F
Iterator::Model<1??P:k@!4Ly??.0@)?3????1?{iy?p?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?۠????@!??a?H?T@)?uoEb?j?1h?L?/?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9u?v?P@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-?\odq@-?\odq@!-?\odq@      ??!       "	Q1??tC?@Q1??tC?@!Q1??tC?@*      ??!       2	h%???¯?h%???¯?!h%???¯?:	?????@?????@!?????@B      ??!       J	???)?j@???)?j@!???)?j@R      ??!       Z	???)?j@???)?j@!???)?j@JGPUYu?v?P@b ?":
functional_3/conv2d_410/Conv2DConv2D?'"??!?'"??":
functional_3/conv2d_478/Conv2DConv2Dl?	?@??!|??Fa??":
functional_3/conv2d_408/Conv2DConv2D????Lҍ?!-x??է?":
functional_3/conv2d_480/Conv2DConv2D'
???L??!??)??":
functional_3/conv2d_481/Conv2DConv2D????ш?!?9???.??":
functional_3/conv2d_407/Conv2DConv2D????l??!??qP???"f
=gradient_tape/functional_3/conv_7b/Conv2D/Conv2DBackpropInputConv2DBackpropInput??L??,??!????ᡷ?"h
>gradient_tape/functional_3/conv_7b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)"????!2ԷF?%??"7
functional_3/conv_7b/Conv2DConv2Dw?B-??!#·Ek??"?
#functional_3/block17_16_conv/Conv2DConv2D?+?bz?!?? ????Q      Y@Y??zJ???a?
kg?X@qg?(??Q @y??2F?M&?"?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 