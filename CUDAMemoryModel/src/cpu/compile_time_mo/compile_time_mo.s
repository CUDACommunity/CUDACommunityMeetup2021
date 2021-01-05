	.file	"compile_time_mo.cpp"
	.intel_syntax noprefix
	.text
	.p2align 4,,15
	.globl	_Z3foov
	.type	_Z3foov, @function
_Z3foov:
.LFB0:
	.cfi_startproc
	mov	eax, DWORD PTR Y[rip]
	add	eax, 1
	mov	DWORD PTR X[rip], eax
	mov	DWORD PTR Y[rip], 0
	ret
	.cfi_endproc
.LFE0:
	.size	_Z3foov, .-_Z3foov
	.globl	Y
	.bss
	.align 4
	.type	Y, @object
	.size	Y, 4
Y:
	.zero	4
	.globl	X
	.align 4
	.type	X, @object
	.size	X, 4
X:
	.zero	4
	.ident	"GCC: (GNU) 7.3.1 20180712 (Red Hat 7.3.1-6)"
	.section	.note.GNU-stack,"",@progbits
