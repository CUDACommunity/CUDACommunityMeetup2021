	.file	"compile_time_mo.cpp"
	.intel_syntax noprefix
	.text
	.p2align 4,,15
	.globl	_Z3foov
	.type	_Z3foov, @function
_Z3foov:
.LFB0:
	.cfi_startproc
	mov	eax, DWORD PTR B[rip]
	mov	DWORD PTR B[rip], 0
	add	eax, 1
	mov	DWORD PTR A[rip], eax
	ret
	.cfi_endproc
.LFE0:
	.size	_Z3foov, .-_Z3foov
	.globl	B
	.bss
	.align 4
	.type	B, @object
	.size	B, 4
B:
	.zero	4
	.globl	A
	.align 4
	.type	A, @object
	.size	A, 4
A:
	.zero	4
	.ident	"GCC: (GNU) 7.3.1 20180712 (Red Hat 7.3.1-6)"
	.section	.note.GNU-stack,"",@progbits
