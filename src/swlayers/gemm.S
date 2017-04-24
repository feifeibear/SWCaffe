	.set noreorder
	.set noat
	.arch sw3
	#  /usr/sw-mpp/swcc/lib/gcc-lib/sw_64-swcc-linux/5.421-sw-482/be::5.421-sw-482

	#-----------------------------------------------------------
	# Compiling gemm.c (/tmp/ccI#.7WCERO)
	#-----------------------------------------------------------

	#-----------------------------------------------------------
	# Options:
	#-----------------------------------------------------------
	#  Target:SW3, ISA:ISA_1, Endian:little, Pointer Size:64
	#  -O3	(Optimization level)
	#  -g0	(Debug level)
	#  -m2	(Report advisories)
	#-----------------------------------------------------------

	.file	1	"/home/export/online1/swyf/swdnn/fjr/swCNNv14-bp/gemm.c"
	.file	2	"/usr/sw-mpp/swcc/lib/gcc-lib/sw_64-swcc-linux/5.421-sw-482/include/simd.h"


	.section .text1, "ax", "progbits"
	.align	4

	.section .bss, "wa", "nobits"
	.align	5

	.section .bss
	.org 0x0
	.align	0
	.type	anon25_107, @object
	.size	anon25_107, 32
anon25_107:	# 0x0
	.skip 32
	.org 0x20
	.align	0
	.type	anon26_108, @object
	.size	anon26_108, 32
anon26_108:	# 0x20
	.skip 32
	.org 0x40
	.align	0
	.type	anon27_109, @object
	.size	anon27_109, 32
anon27_109:	# 0x40
	.skip 32
	.org 0x60
	.align	0
	.type	anon28_110, @object
	.size	anon28_110, 32
anon28_110:	# 0x60
	.skip 32
	.org 0x80
	.align	0
	.type	anon29_111, @object
	.size	anon29_111, 32
anon29_111:	# 0x80
	.skip 32
	.org 0xa0
	.align	0
	.type	anon30_112, @object
	.size	anon30_112, 32
anon30_112:	# 0xa0
	.skip 32
	.org 0xc0
	.align	0
	.type	anon31_113, @object
	.size	anon31_113, 32
anon31_113:	# 0xc0
	.skip 32
	.org 0xe0
	.align	0
	.type	anon32_114, @object
	.size	anon32_114, 32
anon32_114:	# 0xe0
	.skip 32
	.org 0x100
	.align	0
	.type	anon33_115, @object
	.size	anon33_115, 32
anon33_115:	# 0x100
	.skip 32
	.org 0x120
	.align	0
	.type	anon34_116, @object
	.size	anon34_116, 32
anon34_116:	# 0x120
	.skip 32
	.org 0x140
	.align	0
	.type	anon35_117, @object
	.size	anon35_117, 32
anon35_117:	# 0x140
	.skip 32
	.org 0x160
	.align	0
	.type	anon36_118, @object
	.size	anon36_118, 32
anon36_118:	# 0x160
	.skip 32
	.org 0x180
	.align	0
	.type	anon37_119, @object
	.size	anon37_119, 32
anon37_119:	# 0x180
	.skip 32
	.org 0x1a0
	.align	0
	.type	anon38_120, @object
	.size	anon38_120, 32
anon38_120:	# 0x1a0
	.skip 32
	.org 0x1c0
	.align	0
	.type	anon39_121, @object
	.size	anon39_121, 32
anon39_121:	# 0x1c0
	.skip 32
	.org 0x1e0
	.align	0
	.type	anon40_122, @object
	.size	anon40_122, 32
anon40_122:	# 0x1e0
	.skip 32
	.org 0x200
	.align	0
	.type	anon41_123, @object
	.size	anon41_123, 32
anon41_123:	# 0x200
	.skip 32
	.org 0x220
	.align	0
	.type	anon42_124, @object
	.size	anon42_124, 32
anon42_124:	# 0x220
	.skip 32
	.org 0x240
	.align	0
	.type	anon43_125, @object
	.size	anon43_125, 32
anon43_125:	# 0x240
	.skip 32
	.org 0x260
	.align	0
	.type	anon44_126, @object
	.size	anon44_126, 32
anon44_126:	# 0x260
	.skip 32
	.section .text1

	# Program Unit: gemm
	.align 4
	.ent	gemm#
	.globl	gemm
gemm:	# 0x0
	# anon13 = 544
	# anon14 = 560
	# anon15 = 548
	# anon16 = 576
	# anon17 = 592
	# anon19 = 736
	# anon20 = 608
	# anon21 = 32
	# anon22 = 568
	# anon23 = 584
	# anon24 = 552
	# return_address = 0
	# _temp_gra_spill0 = 864
	# _temp_gra_spill1 = 872
	# _temp_gra_spill2 = 880
	# _temp_gra_spill3 = 888
	# _temp_gra_spill4 = 896
	.loc	1	23	0
#  19  //cN - cNo
#  20  //cK - cNi
#  21  //
#  22  //M*K K*N
#  23  void gemm(Type* input, Type* weight, Type* output, int M, int Mld, int N, int K, int rid, int cid){
.BB1_gemm:
	.prologue
	ldih	$gp,0($27)               	!gpdisp!1	# [0] 0
	ldi	$gp,0($gp)                	!gpdisp!1	# [1] 0
	
$ng..gemm:
	ldi	$sp,-960($sp)             	# [3] 
	.loc	1	31	0
#  27  	SIMDType tmp_weight[4];
#  28      SIMDType tmp_input[4];
#  29  	SIMDType tmp_output[16];
#  30  
#  31      for(ccCore=0; ccCore<8; ++ccCore){
	.frame $15,960,$26,0
	.mask 0x4008000,-960
	stl	$26, 0($sp)
	vinsw	$31,$29,7,$29           # [3] ccCore = 0 
	mov	$31,$5						# ccCore
	mov	$21,$29						#	N
	vinsw	$19,$29,2,$29			# M
	s8addl	$20,$31,$0
	vinsw	$0,$29,3,$29			# 8*Mld
	.loc	1	31	0
	.loc	1	23	0
	stl	$18,928($sp)              	# [6] output
	vinsf	$17,$19,2,$19			# weight
	.loc	1	31	0
	.loc	1	23	0
	vinsf	$16,$19,3,$19			# input

	vinsw	$13,$29,6,$29
	stl	$19,896($sp)              	# [8] _temp_gra_spill4
	stl	$9,872($sp)               	# [9] _temp_gra_spill1
	ldw	$19, 960($sp)

	br	$31,.begin_loop # [9] 
	.align	4

.begin_loop:
	ldw		$6,976($sp)               	# [0] $0 = cid
	cmpeq	$6,$5,$6            	# [3] cid == ccCore ? 
	ldw		$0,968($sp)               	# [2] rid
	cmpeq	$0,$5,$0                	# [0] rid == ccCore
	beq		$6,.inLoop_ccCore_Getr # [4] else 
	br		$31,.inLoop_ccCore_Putr
	.align	4

.inLoop_ccCore_Getr:
	beq		$0,.inLoop_ccCore_Getr_Getc		# [1] else 
	br		$31,.inLoop_ccCore_Getr_Putc
	.align	4

.inLoop_ccCore_Putr:
	beq		$0,.inLoop_ccCore_Putr_Getc		# [1] else 
	br		$31,.inLoop_ccCore_Putr_Putc
	.align	4
###############################
###########SECTION1############
###############################
.inLoop_ccCore_Getr_Getc:
	vinsw	$31,$29,6,$29			# cM = 0

.inLoop_cM_ccCore_Getr_Getc:

	ldl		$5,928($sp)              	# [0] $5 = output

	vextw	$29,6,$0
	s8addl	$0,$31,$0             	# [0] $0 = cM*8 
	s4addl	$0,$5,$0              	# [3] $0 = output_ptr = cM*32+output  
	vinsw	$31,$29,5,$29			# cN

.inLoop_cN_ccCore_Getr_Getc:

### begin 16 ldw stw

		vinsf	$0,$19,1,$19				# output_ptr
	vldd	$11, 0($0)
		addw	$31,2,$28
	getr	$1
		vextw	$29,2,$5				# M
	getc	$7                       	# [1] 

		s8addl	$5,$31,$5            	# [0] $5 = 8*M 
	vldd	$12, 32($0)
		s4addl  $5, $0, $25
	getr	$2
		s4addl  $5, $25, $26
	getc	$8                       	# [1] 

	vldd	$13, 64($0)
		vmad	$1,$7,$11,$11
	getr	$3
		s4addl  $5, $26, $27
	getc	$9                       	# [1] 

	vldd	$14, 96($0)
		vmad	$2,$7,$12,$12
	getr	$4
	getc	$10                       	# [1] 
		vmad	$3,$7,$13,$13
	

	vldd	$15, 0($25)
	vldd	$16, 32($25)
	vldd	$17, 64($25)
		vmad	$4,$7,$14,$14
	vldd	$18, 96($25)
		getc	$7                       	# [1] 


	vldd	$20, 0($26)
		vmad	$1,$8,$15,$15
	vldd	$21, 32($26)
		vmad	$2,$8,$16,$16
	vldd	$22, 64($26)
		vmad	$3,$8,$17,$17
	vldd	$23, 96($26)
		vmad	$4,$8,$18,$18


	vldd	$24, 0($27)
		vmad	$1,$9,$20,$20
	vldd	$25, 32($27)
		vmad	$2,$9,$21,$21
	vldd	$26, 64($27)
		vmad	$3,$9,$22,$22
	vldd	$27, 96($27)
		vmad	$4,$9,$23,$23

	vmad	$1,$10,$24,$24
		getr	$1
	vmad	$2,$10,$25,$25
		getr	$2
	vmad	$3,$10,$26,$26
		getr	$3
	vmad	$4,$10,$27,$27
		getr	$4




.inLoop_cK_ccCore_Getr_Getc:
#.core_compute # [1] 


	vmad	$1,$7,$11,$11
		getc	$8                       	# [1] 
	vmad	$2,$7,$12,$12
		getc	$9                       	# [1] 
	vmad	$3,$7,$13,$13
		getc	$10                       	# [1] 
	vmad	$4,$7,$14,$14
	
	vmad	$1,$8,$15,$15
	vmad	$2,$8,$16,$16
		addw	$28,1,$28                	# [7] $5 = cK+=1
	vmad	$3,$8,$17,$17
	vmad	$4,$8,$18,$18

	vmad	$1,$9,$20,$20
	vmad	$2,$9,$21,$21
		cmple	$19,$28,$0              	# [8] $19 = K <= cK 
	vmad	$3,$9,$22,$22
	vmad	$4,$9,$23,$23
		getc	$7                       	# [1] 

	vmad	$1,$10,$24,$24
		getr	$1
	vmad	$2,$10,$25,$25
		getr	$2
	vmad	$3,$10,$26,$26
		getr	$3
	vmad	$4,$10,$27,$27
		getr	$4

	beq		$0,.inLoop_cK_ccCore_Getr_Getc				# [9] ! K > cK 

#.exit_cK_ccCore_Getr_Getc:

	vextw	$29,2,$5				# M
	vextf	$19,1,$6				# output_ptr

	vmad	$1,$7,$11,$11
	vmad	$2,$7,$12,$12
		getc	$8                       	# [1] 
	vmad	$3,$7,$13,$13
		getc	$9                       	# [1] 
	vmad	$4,$7,$14,$14
		getc	$10                       	# [1] 
	
	vmad	$1,$8,$15,$15
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 
	vmad	$2,$8,$16,$16
		s4addl  $0, $6, $7
	vmad	$3,$8,$17,$17
		s4addl  $0, $7, $5
	vmad	$4,$8,$18,$18
		s4addl  $0, $5, $8

	vmad	$1,$9,$20,$20
		vstd	$11, 0($6)
	vmad	$2,$9,$21,$21
		vstd	$12, 32($6)
	vmad	$3,$9,$22,$22
		vstd	$13, 64($6)
	vmad	$4,$9,$23,$23
		vstd	$14, 96($6)

	vmad	$1,$10,$24,$24
		vstd	$15, 0($7)
	vmad	$2,$10,$25,$25
		vstd	$16, 32($7)
	vmad	$3,$10,$26,$26
		vstd	$17, 64($7)
	vmad	$4,$10,$27,$27
		vstd	$18, 96($7)

	vstd	$20, 0($5)
		vextw	$29,2,$1				# M
	vstd	$21, 32($5)
		sll		$1,7,$6                  	# [4] $6 = M*128 
	vstd	$22, 64($5)
		vextf	$19,1,$0				# output_ptr
	vstd	$23, 96($5)
		addl	$0,$6,$0                 	# [0] output_ptr _  

	vstd	$24, 0($8)
		vextw	$29,5,$1
	vstd	$25, 32($8)
		addw	$1,4,$1                	# [4] cN+=4 
	vstd	$26, 64($8)
		vinsw	$1,$29,5,$29			
	vstd	$27, 96($8)
		cmplt	$1,$29,$6              	# [0] $5 = cN < N 

	bne	$6,.inLoop_cN_ccCore_Getr_Getc # [0] exit loop 

#.exit_cN:
	vextw	$29,6,$0
	addw	$0,4,$0                	# [0] $0 = cM 
	vinsw	$0,$29,6,$29
	vextw	$29,2,$5				# M
	cmplt	$0,$5,$6              	# [1] $5 = M 
	bne		$6,.inLoop_cM_ccCore_Getr_Getc			# [2] cM > M exit

#.exit_Loop:
	.loc	1	31	0
	vextw	$29,7,$5
	addw	$5,1,$5                  	# [0] ccCore++ 
	vinsw	$5,$29,7,$29
	cmple	$5,7,$0                 	# [1] ccCore<=7 
	bne	$0,.begin_loop					# [2] 
	br	$31,.exit_program
###############################
###########SECTION2############
###############################
.inLoop_ccCore_Getr_Putc:
	vinsw	$31,$29,6,$29			# cM = 0

.inLoop_cM_ccCore_Getr_Putc:
	ldl	$5,928($sp)              	# [0] $5 = output

	vextw	$29,6,$0
	s8addl	$0,$31,$0             	# [0] $0 = cM*8 
	vinsw	$31,$29,5,$29			# cN
	s4addl	$0,$5,$0              	# [3] $0 = output_ptr = cM*32+output  

.inLoop_cN_ccCore_Getr_Putc:


		vextw	$29,2,$21				# M
		vextw	$29,5,$5	
	s8addl	$21,$31,$21            	# [0] $5 = 8*M 
		vextf	$19,2,$6
	mulw	$5, $19, $5				# K*cN*8
		vinsf	$0,$19,1,$19				# output_ptr
	s8addl	$5,$6,$6              	# [3] weight_ptr += weight+8*cN 

	vldd	$11, 0($0)
	getr	$1
	lddec	$7,0($6)
		s8addl	$19, $6, $22


	vldd	$12, 32($0)
		s4addl  $21, $0, $25
	getr	$2
		s4addl  $21, $25, $26
	lddec	$8,0($22)
		s8addl	$19, $22, $23

	vldd	$13, 64($0)
		vmad	$1,$7,$11,$11
	getr	$3
		s4addl  $21, $26, $27
	lddec	$9, 0($23)
		s8addl	$19, $23, $24

	vldd	$14, 96($0)
		vmad	$2,$7,$12,$12
	getr	$4
		addl	$6, 8 ,$6              	# [4] weight_ptr += 8  
	lddec	$10, 0($24)
		vmad	$3,$7,$13,$13


	vldd	$15, 0($25)
		addw	$31,2,$28				# cNi
	vldd	$16, 32($25)
	vldd	$17, 64($25)
		vmad	$4,$7,$14,$14
	vldd	$18, 96($25)
		lddec	$7,0($6)

	vldd	$20, 0($26)
		vmad	$1,$8,$15,$15
	vldd	$21, 32($26)
		vmad	$2,$8,$16,$16
	vldd	$22, 64($26)
		vmad	$3,$8,$17,$17
	vldd	$23, 96($26)
		vmad	$4,$8,$18,$18

	vldd	$24, 0($27)
		vmad	$1,$9,$20,$20
	vldd	$25, 32($27)
		vmad	$2,$9,$21,$21
	vldd	$26, 64($27)
		vmad	$3,$9,$22,$22
	vldd	$27, 96($27)
		vmad	$4,$9,$23,$23

	vmad	$1,$10,$24,$24
		getr	$1
	vmad	$2,$10,$25,$25
		getr	$2
	vmad	$3,$10,$26,$26
		getr	$3
	vmad	$4,$10,$27,$27
		getr	$4


.inLoop_cK_ccCore_Getr_Putc:

	vmad	$1,$7,$11,$11
		s8addl	$19, $6, $0
	vmad	$2,$7,$12,$12
		lddec	$8,0($0)
	vmad	$3,$7,$13,$13
		s8addl	$19, $0, $0
	vmad	$4,$7,$14,$14
		lddec	$9, 0($0)
	
	vmad	$1,$8,$15,$15
		s8addl	$19, $0, $0
	vmad	$2,$8,$16,$16
		lddec	$10, 0($0)
	vmad	$3,$8,$17,$17
		addw	$28,1,$28                	# [7] $5 = cK+=1
	vmad	$4,$8,$18,$18
		cmple	$19,$28,$0              	# [8] $19 = K <= cK 

		addl	$6, 8 ,$6              	# [4] weight_ptr += 8  
	vmad	$1,$9,$20,$20
		lddec	$7,0($6)
	vmad	$2,$9,$21,$21
	vmad	$3,$9,$22,$22
	vmad	$4,$9,$23,$23

	vmad	$1,$10,$24,$24
		getr	$1
	vmad	$2,$10,$25,$25
		getr	$2
	vmad	$3,$10,$26,$26
		getr	$3
	vmad	$4,$10,$27,$27
		getr	$4
	beq		$0,.inLoop_cK_ccCore_Getr_Putc				# [9] ! K > cK 


#.exit_cK_ccCore_Getr_Putc:

	vextw	$29,2,$5				# M
		s8addl	$19, $6, $0

	vmad	$1,$7,$11,$11
		lddec	$8,0($0)
	vmad	$2,$7,$12,$12
		s8addl	$19, $0, $0
	vmad	$3,$7,$13,$13
		lddec	$9, 0($0)
	vmad	$4,$7,$14,$14
		s8addl	$19, $0, $0
	
	vmad	$1,$8,$15,$15
		lddec	$10, 0($0)
	vmad	$2,$8,$16,$16
		vextf	$19,1,$6				# output_ptr
	vmad	$3,$8,$17,$17
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 
	vmad	$4,$8,$18,$18
		s4addl  $0, $6, $7

	vmad	$1,$9,$20,$20
		vstd	$11, 0($6)
	vmad	$2,$9,$21,$21
		vstd	$12, 32($6)
	vmad	$3,$9,$22,$22
		vstd	$13, 64($6)
	vmad	$4,$9,$23,$23
		vstd	$14, 96($6)


	vmad	$1,$10,$24,$24
		vstd	$15, 0($7)
	vmad	$2,$10,$25,$25
		vstd	$16, 32($7)
	vmad	$3,$10,$26,$26
		vstd	$17, 64($7)
	vmad	$4,$10,$27,$27
		vstd	$18, 96($7)

	s8addl	$5,$31,$0            	# [0] $5 = 8*M 
		s4addl  $0, $7, $12

	vstd	$20, 0($12)
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 
	vstd	$21, 32($12)
		s4addl  $0, $12, $13
	vstd	$22, 64($12)
		sll		$5,7,$6                  	# [4] $6 = M*128 
	vstd	$23, 96($12)
		vextf	$19,1,$0				# output_ptr


	vstd	$24, 0($13)
		addl	$0,$6,$0                 	# [0] output_ptr _  
	vstd	$25, 32($13)
		vextw	$29,5,$5
	vstd	$26, 64($13)
		addw	$5,4,$5                	# [4] cN+=4 
	vstd	$27, 96($13)
		vinsw	$5,$29,5,$29
	cmplt	$5,$29,$6              	# [0] $5 = cN < N 

	bne	$6,.inLoop_cN_ccCore_Getr_Putc # [0] exit loop 

#.exit_cN:
	vextw	$29,6,$0
	addw	$0,4,$0                	# [0] $0 = cM 
	vinsw	$0,$29,6,$29
	vextw	$29,2,$5				# M
	cmplt	$0,$5,$6              	# [1] $5 = M 
	bne		$6,.inLoop_cM_ccCore_Getr_Putc			# [2] cM > M exit

#.exit_Loop:
	.loc	1	31	0
	vextw	$29,7,$5
	addw	$5,1,$5                  	# [0] ccCore++ 
	vinsw	$5,$29,7,$29
	cmple	$5,7,$0                 	# [1] ccCore<=7 
	bne	$0,.begin_loop					# [2] 
	br	$31,.exit_program
###############################
###########SECTION3############
###############################
.inLoop_ccCore_Putr_Getc:
	vinsw	$31,$29,6,$29			# cM = 0

.inLoop_cM_ccCore_Putr_Getc:
	ldl	$5,928($sp)              	# [0] $5 = output

	vextw	$29,6,$0
	s8addl	$0,$31,$0             	# [0] $0 = cM*8 
	vinsw	$31,$29,5,$29			# cN
	s4addl	$0,$5,$0              	# [3] $0 = output_ptr = cM*32+output  

.inLoop_cN_ccCore_Putr_Getc:

		vextw	$29,2,$24				# M
	
		vextf	$19,3,$5
	vldd	$11, 0($0)
		vextw	$29,6,$23
	sll		$23, 5, $23				
		addw	$23, $5, $5				# input + 4*cM
	vldr	$1, 0($5)
	getc	$7                       	# [1] 

	vldd	$12, 32($0)
		s8addl	$24,$31,$24            	# [0] $5 = 8*M 
	vldr	$2, 32($5)              	# [3] id:218
		vinsf	$0,$19,1,$19				# output_ptr
	getc	$8                       	# [1] 
		s4addl  $24, $0, $25

	vldd	$13, 64($0)
		vmad	$1,$7,$11,$11
	vldr	$3, 64($5)              	# [3] id:218
		s4addl  $24, $25, $26
	getc	$9                       	# [1] 
		s4addl  $24, $26, $27

	vldd	$14, 96($0)
		vmad	$2,$7,$12,$12
  #bug for acc_trans
	vldr	$4, 96($5)              	# [3] id:218
		vextw	$29,3,$6			# Mld
	getc	$10                       	# [1] 
		vmad	$3,$7,$13,$13
	
	vldd	$15, 0($25)
		addw	$31,2,$28					# cK
	vldd	$16, 32($25)
		s4addl	$6,$5,$5            	# [7] input = 4*8*Mld+input 
	vldd	$17, 64($25)
		vmad	$4,$7,$14,$14
	vldd	$18, 96($25)
		vmad	$1,$8,$15,$15

	vldd	$20, 0($26)
		vmad	$2,$8,$16,$16
	vldd	$21, 32($26)
		getc	$7                       	# [1] 
	vldd	$22, 64($26)
		vmad	$3,$8,$17,$17
	vldd	$23, 96($26)
		vmad	$4,$8,$18,$18

	vldd	$24, 0($27)
		vmad	$1,$9,$20,$20
	vldd	$25, 32($27)
		vmad	$2,$9,$21,$21
	vldd	$26, 64($27)
		vmad	$3,$9,$22,$22
	vldd	$27, 96($27)
		vmad	$4,$9,$23,$23

	vmad	$1,$10,$24,$24
		vldr	$1, 0($5)
	vmad	$2,$10,$25,$25
		vldr	$2, 32($5)              	# [3] id:218
	vmad	$3,$10,$26,$26
		vldr	$3, 64($5)              	# [3] id:218
	vmad	$4,$10,$27,$27
		vldr	$4, 96($5)              	# [3] id:218

		s4addl	$6,$5,$5            	# [7] input = 4*8*Mld+input 



.inLoop_cK_ccCore_Putr_Getc:

	vmad	$1,$7,$11,$11
		getc	$8                       	# [1] 
	vmad	$2,$7,$12,$12
	vmad	$3,$7,$13,$13
		getc	$9                       	# [1] 
	vmad	$4,$7,$14,$14
	vmad	$1,$8,$15,$15
		getc	$10                       	# [1] 
	vmad	$2,$8,$16,$16
		addw	$28,1,$28                	# [7] $5 = cK+=1
	vmad	$3,$8,$17,$17
		cmple	$19,$28,$0              	# [8] $19 = K <= cK 
	vmad	$4,$8,$18,$18

	vmad	$1,$9,$20,$20
	vmad	$2,$9,$21,$21
	vmad	$3,$9,$22,$22
		getc	$7                       	# [1] 
	vmad	$4,$9,$23,$23

	vmad	$1,$10,$24,$24
		vldr	$1, 0($5)
	vmad	$2,$10,$25,$25
		vldr	$2, 32($5)              	# [3] id:218
	vmad	$3,$10,$26,$26
		vldr	$3, 64($5)              	# [3] id:218
	vmad	$4,$10,$27,$27
		vldr	$4, 96($5)              	# [3] id:218
	s4addl	$6,$5,$5            	# [7] input = 4*8*Mld+input 


	beq		$0,.inLoop_cK_ccCore_Putr_Getc				# [9] ! K > cK 

#.exit_cK_ccCore_Putr_Getc:

	vmad	$1,$7,$11,$11
		getc	$8                       	# [1] 
	vmad	$2,$7,$12,$12
		getc	$9                       	# [1] 
	vmad	$3,$7,$13,$13
		getc	$10                       	# [1] 
	vmad	$4,$7,$14,$14
		vextf	$19,1,$6				# output_ptr
	

	vmad	$1,$8,$15,$15
		vextw	$29,2,$5				# M
	vmad	$2,$8,$16,$16
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 
	vmad	$3,$8,$17,$17
		s4addl  $0, $6, $7
	vmad	$4,$8,$18,$18
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 

	vmad	$1,$9,$20,$20
		vstd	$11, 0($6)
	vmad	$2,$9,$21,$21
		vstd	$12, 32($6)
	vmad	$3,$9,$22,$22
		vstd	$13, 64($6)
	vmad	$4,$9,$23,$23
		vstd	$14, 96($6)

	vmad	$1,$10,$24,$24
		vstd	$15, 0($7)
	vmad	$2,$10,$25,$25
		vstd	$16, 32($7)
	vmad	$3,$10,$26,$26
		vstd	$17, 64($7)
	vmad	$4,$10,$27,$27
		vstd	$18, 96($7)

	s4addl  $0, $7, $8
		vstd	$20, 0($8)
	s8addl	$5,$31,$0            	# [0] $5 = 8*M 
		vstd	$21, 32($8)
	s4addl  $0, $8, $9
		vstd	$22, 64($8)
	sll		$5,7,$6                  	# [4] $6 = M*128 
		vstd	$23, 96($8)
	vextf	$19,1,$0				# output_ptr
		vstd	$24, 0($9)
	addl	$0,$6,$0                 	# [0] output_ptr _  
		vstd	$25, 32($9)
	vextw	$29,5,$5
		vstd	$26, 64($9)
	addw	$5,4,$5                	# [4] cN+=4 
		vstd	$27, 96($9)
	vinsw	$5,$29,5,$29
		cmplt	$5,$29,$6              	# [0] $5 = cN < N 
	bne	$6,.inLoop_cN_ccCore_Putr_Getc # [0] exit loop 

#.exit_cN:
	vextw	$29,6,$0
	addw	$0,4,$0                	# [0] $0 = cM 
	vinsw	$0,$29,6,$29
	vextw	$29,2,$5				# M
	cmplt	$0,$5,$6              	# [1] $5 = M 
	bne		$6,.inLoop_cM_ccCore_Putr_Getc			# [2] cM > M exit

#.exit_Loop:
	.loc	1	31	0
	vextw	$29,7,$5
	addw	$5,1,$5                  	# [0] ccCore++ 
	vinsw	$5,$29,7,$29
	cmple	$5,7,$0                 	# [1] ccCore<=7 
	bne	$0,.begin_loop					# [2] 
	br	$31,.exit_program
###############################
###########SECTION4############
###############################
.inLoop_ccCore_Putr_Putc:
	vinsw	$31,$29,6,$29			# cM = 0

.inLoop_cM_ccCore_Putr_Putc:
	ldl	$5,928($sp)              	# [0] $5 = output

	vextw	$29,6,$0
	s8addl	$0,$31,$0             	# [0] $0 = cM*8 
	vinsw	$31,$29,5,$29			# cN
	s4addl	$0,$5,$0              	# [3] $0 = output_ptr = cM*32+output  

.inLoop_cN_ccCore_Putr_Putc:

### begin 16 ldw stw

	mov		$0,$24
		vextw	$29,2,$5				# M
	s8addl	$5,$31,$5            	# [0] $5 = 8*M 
		vextf	$19,2,$6
	s4addl  $5, $0, $25
		vextw	$29,5,$22
	s4addl  $5, $25, $26
		mulw	$22, $19, $22				# K*cN*8
	s4addl  $5, $26, $27
		vextw	$29,6,$23

		s8addl	$22,$6,$6              	# [3] weight_ptr += weight+8*cN 
	vextf	$19,3,$5
		vinsf	$0,$19,1,$19				# output_ptr
	sll		$23, 5, $23				
		addw	$23, $5, $5				# input + 4*cM

	vldd	$11, 0($24)
	vldr	$1, 0($5)
	lddec	$7,0($6)
		s8addl	$19, $6, $0

	vldd	$12, 32($24)
		vmad	$1,$7,$11,$11
	vldr	$2, 32($5)              	# [3] id:218
	lddec	$8,0($0)
		s8addl	$19, $0, $0

	vldd	$13, 64($24)
		vextw	$29,3,$23		# Mld
	vldr	$3, 64($5)              	# [3] id:218
	lddec	$9, 0($0)
		s8addl	$19, $0, $0

	vldd	$14, 96($24)
		vmad	$2,$7,$12,$12
	vldr	$4, 96($5)              	# [3] id:218
		addl	$6, 8 ,$6              	# [4] weight_ptr += 8  
	lddec	$10, 0($0)
		vmad	$3,$7,$13,$13
	

	vldd	$15, 0($25)
		addw	$31,2,$28					# cK
	vldd	$16, 32($25)
		vmad	$4,$7,$14,$14
	vldd	$17, 64($25)
		lddec	$7,0($6)
	vldd	$18, 96($25)
		s4addl	$23,$5,$5            	# [7] input = 4*8*Mld+input 


	vldd	$20, 0($26)
		vmad	$1,$8,$15,$15
	vldd	$21, 32($26)
		vmad	$2,$8,$16,$16
	vldd	$22, 64($26)
		vmad	$3,$8,$17,$17
	vldd	$23, 96($26)
		vmad	$4,$8,$18,$18


	vldd	$24, 0($27)
		vmad	$1,$9,$20,$20
	vldd	$25, 32($27)
		vmad	$2,$9,$21,$21
	vldd	$26, 64($27)
		vmad	$3,$9,$22,$22
	vldd	$27, 96($27)
		vmad	$4,$9,$23,$23


	vmad	$1,$10,$24,$24
		vldr	$1, 0($5)
	vmad	$2,$10,$25,$25
		vldr	$2, 32($5)              	# [3] id:218
	vmad	$3,$10,$26,$26
		vldr	$3, 64($5)              	# [3] id:218
	vmad	$4,$10,$27,$27
		vldr	$4, 96($5)              	# [3] id:218

	vextw	$29,3,$0			# Mld
		s4addl	$0,$5,$5            	# [7] input = 4*8*Mld+input 

.inLoop_cK_ccCore_Putr_Putc:

		s8addl	$19, $6, $0
	vmad	$1,$7,$11,$11
		lddec	$8,0($0)
	vmad	$2,$7,$12,$12
		s8addl	$19, $0, $0
	vmad	$3,$7,$13,$13
		lddec	$9, 0($0)
	vmad	$4,$7,$14,$14
	
		s8addl	$19, $0, $0
	vmad	$1,$8,$15,$15
		lddec	$10, 0($0)
	vmad	$2,$8,$16,$16
		addw	$28,1,$28               	# [7] $5 = cK+=1
	vmad	$3,$8,$17,$17
		addl	$6, 8 ,$6              	# [4] weight_ptr += 8  
	vmad	$4,$8,$18,$18
		lddec	$7,0($6)

	vmad	$1,$9,$20,$20
		cmple	$19,$28,$0              	# [8] $19 = K <= cK 
	vmad	$2,$9,$21,$21
	vmad	$3,$9,$22,$22
	vmad	$4,$9,$23,$23

	vmad	$1,$10,$24,$24
		vldr	$1, 0($5)
	vmad	$2,$10,$25,$25
		vldr	$2, 32($5)              	# [3] id:218
	vmad	$3,$10,$26,$26
		vldr	$3, 64($5)              	# [3] id:218
	vmad	$4,$10,$27,$27
		vldr	$4, 96($5)              	# [3] id:218
	vextw	$29,3,$8			# Mld
		s4addl	$8,$5,$5            	# [7] input = 4*8*Mld+input 

	beq		$0,.inLoop_cK_ccCore_Putr_Putc				# [9] ! K > cK 


#.exit_cK_ccCore_Putr_Putc:

		s8addl	$19, $6, $0
	vmad	$1,$7,$11,$11
		lddec	$8,0($0)
	vmad	$2,$7,$12,$12
		s8addl	$19, $0, $0
	vmad	$3,$7,$13,$13
		lddec	$9, 0($0)
	vmad	$4,$7,$14,$14
		s8addl	$19, $0, $0
	vextw	$29,2,$5				# M
		vextf	$19,1,$6				# output_ptr

	vmad	$1,$8,$15,$15
		lddec	$10, 0($0)
	vmad	$2,$8,$16,$16
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 
	vmad	$3,$8,$17,$17
	vmad	$4,$8,$18,$18


	vmad	$1,$9,$20,$20
		vstd	$11, 0($6)
	vmad	$2,$9,$21,$21
		vstd	$12, 32($6)
	vmad	$3,$9,$22,$22
		vstd	$13, 64($6)
	vmad	$4,$9,$23,$23
		vstd	$14, 96($6)
	s4addl  $0, $6, $11

	vmad	$1,$10,$24,$24
		vstd	$15, 0($11)
	vmad	$2,$10,$25,$25
		vstd	$16, 32($11)
	vmad	$3,$10,$26,$26
		vstd	$17, 64($11)
	vmad	$4,$10,$27,$27
		vstd	$18, 96($11)

	s8addl	$5,$31,$7            	# [0] $5 = 8*M 
		s4addl  $7, $11, $12
	vstd	$20, 0($12)
		s8addl	$5,$31,$0            	# [0] $5 = 8*M 
	vstd	$21, 32($12)
		s4addl  $0, $12, $13
	vstd	$22, 64($12)
		sll		$5,7,$6                  	# [4] $6 = M*128 
	vstd	$23, 96($12)
		vextf	$19,1,$0				# output_ptr

	vstd	$24, 0($13)
		addl	$0,$6,$0                 	# [0] output_ptr _  
	vstd	$25, 32($13)
		vextw	$29,5,$5
	vstd	$26, 64($13)
		addw	$5,4,$5                	# [4] cN+=4 
	vstd	$27, 96($13)
		vinsw	$5,$29,5,$29
	cmplt	$5,$29,$6              	# [0] $5 = cN < N 

	bne	$6,.inLoop_cN_ccCore_Putr_Putc # [0] exit loop 

#.exit_cN:
	vextw	$29,6,$0
	addw	$0,4,$0                	# [0] $0 = cM 
	vinsw	$0,$29,6,$29
	vextw	$29,2,$5				# M
	cmplt	$0,$5,$6              	# [1] $5 = M 
	bne		$6,.inLoop_cM_ccCore_Putr_Putc			# [2] cM > M exit

#.exit_Loop:
	.loc	1	31	0
	vextw	$29,7,$5	
	addw	$5,1,$5                  	# [0] ccCore++ 
	vinsw	$5,$29,7,$29
	cmple	$5,7,$0                 	# [1] ccCore<=7 
	bne	$0,.begin_loop					# [2] 
	br	$31,.exit_program

###############################
##########END   SECTION########
###############################
.exit_program:
	.loc	1	31	0
	ldl	$26, 0($sp)
	ldl	$9,872($sp)               	# [0] _temp_gra_spill1
	ldl	$29,896($sp)              	# [1] _temp_gra_spill4
	ldi	$sp,960($sp)              	# [2] 
	ret	$31,($26),1               	# [2] 
	.align	4

.L_CC_gemm:
# PU cycle count: 0.000000
	.end	gemm
	.section .text1
	.align	4
	.section .bss
	.align	5
#	.gpvalue 0

	.section .debug_info, "", "progbits"
	.align	0
	.byte	0xa6, 0x00, 0x00, 0x00, 0x02, 0x00
	.long	.debug_abbrev
	.long	0x65670108, 0x632e6d6d, 0x65706f00, 0x2043436e
	.byte	0x35, 0x2e, 0x34, 0x32, 0x31, 0x2d, 0x73, 0x77
	.byte	0x2d, 0x34, 0x38, 0x32, 0x00, 0x01, 0x00
	.long	.debug_line
	.byte	0x02, 0x01, 0x17, 0x67, 0x65, 0x6d, 0x6d, 0x00
	.byte	0x01, 0x04, 0x92, 0x1e, 0xc0, 0x07
	.quad	.BB1_gemm
	.quad	.L_CC_gemm
	.long	0x69170103, 0x7475706e, 0x50910200, 0x77170103
	.long	0x68676965, 0x91020074, 0x17010358, 0x7074756f
	.long	0x02007475, 0x01036091, 0x02004d17, 0x01036891
	.long	0x646c4d17, 0x70910200, 0x4e170103, 0x78910200
	.long	0x4b170103, 0x00910200, 0x72170103, 0x02006469
	.byte	0x91, 0x08, 0x03, 0x01, 0x17, 0x63, 0x69, 0x64
	.byte	0x00, 0x02, 0x91, 0x10, 0x00, 0x00, 0x00

	.section .debug_frame, "", "progbits"
	.align	0

	.section .debug_aranges, "", "progbits"
	.align	0
	.byte	0x2c, 0x00, 0x00, 0x00, 0x02, 0x00
	.long	.debug_info
	.byte	0x08, 0x00, 0x00, 0x00, 0x00, 0x00
	.quad	.BB1_gemm
	.quad	.L_CC_gemm - .BB1_gemm
	.long	0x00000000, 0x00000000, 0x00000000, 0x00000000

	.section .debug_pubnames, "", "progbits"
	.align	0
	.byte	0x17, 0x00, 0x00, 0x00, 0x02, 0x00
	.long	.debug_info
	.long	0x000000aa, 0x0000002d, 0x6d6d6567, 0x00000000
	.byte	0x00

	.section .eh_frame, "a", "progbits"
	.align	0

.LEHCIE:
	.long	.LEHCIE_end - .LEHCIE_begin
.LEHCIE_begin:
	.long 0x0
	.byte	0x01, 0x00, 0x01, 0x78, 0x1a, 0x0c, 0x1e, 0x00
	.align 3
.LEHCIE_end:

	.section .debug_abbrev, "", "progbits"
	.align	0
	.long	0x03011101, 0x13082508, 0x100b420b, 0x02000006
	.long	0x0b3a012e, 0x08030b3b, 0x0a400c3f, 0x01120111
	.long	0x05030000, 0x3b0b3a00, 0x0208030b, 0x0000000a
	.byte	0x00
	.section	.note.GNU-stack,"",@progbits
	.ident	"#SWCC Version 5.421-sw-482 : gemm.c compiled with : -O3 -msimd "

