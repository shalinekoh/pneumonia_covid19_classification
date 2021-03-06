?	?c??=;?@?c??=;?@!?c??=;?@	?n??T@?n??T@!?n??T@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?c??=;?@??yȔ??1??'`??@AKY?8?ŝ?I??(??r@Y:x&4?	?@*	?Mb?JS?A2P
Iterator::Model::PrefetchCT??	?@!???:??X@)CT??	?@1???:??X@:Preprocessing2F
Iterator::Model?X???	?@!      Y@)??	/????1??c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 83.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?n??T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??yȔ????yȔ??!??yȔ??      ??!       "	??'`??@??'`??@!??'`??@*      ??!       2	KY?8?ŝ?KY?8?ŝ?!KY?8?ŝ?:	??(??r@??(??r@!??(??r@B      ??!       J	:x&4?	?@:x&4?	?@!:x&4?	?@R      ??!       Z	:x&4?	?@:x&4?	?@!:x&4?	?@JGPUY?n??T@b ?":
functional_3/conv2d_410/Conv2DConv2D????$??!????$??":
functional_3/conv2d_480/Conv2DConv2De?ʵ)M??!{NП8??":
functional_3/conv2d_408/Conv2DConv2D?J?7??!??
???":
functional_3/conv2d_407/Conv2DConv2D??r??!X??&???":
functional_3/conv2d_479/Conv2DConv2D??8`????!???R5???":
functional_3/conv2d_478/Conv2DConv2D5F[????!U?>?j0??"7
functional_3/conv_7b/Conv2DConv2D#???????!X8R,??">
"functional_3/block17_1_conv/Conv2DConv2D7?\d???!%~x??":
functional_3/conv2d_567/Conv2DConv2D?-???d??!?'?_???":
functional_3/conv2d_481/Conv2DConv2D??V跜?!.?k?ݾ??Q      Y@Y?'??4??a[?j??X@q^0F???u?y?[?S??"?
host?Your program is HIGHLY input-bound because 83.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 