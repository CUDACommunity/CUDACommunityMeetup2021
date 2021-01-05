
Fatbin elf code:
================
arch = sm_52
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

	code for sm_52
		Function : _Z3fooPii
	.headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"
                                                                                     /* 0x001fc400fe2007f6 */
        /*0008*/                   MOV R1, c[0x0][0x20] ;                            /* 0x4c98078000870001 */
        /*0010*/                   MOV R18, c[0x0][0x140] ;                          /* 0x4c98078005070012 */
        /*0018*/                   MOV R19, c[0x0][0x144] ;                          /* 0x4c98078005170013 */
                                                                                     /* 0x001fc801fe2007f6 */
        /*0028*/                   MOV32I R0, 0xffffff9c ;                           /* 0x010ffffff9c7f000 */
        /*0030*/                   MOV R2, R18 ;                                     /* 0x5c98078001270002 */
        /*0038*/                   MOV R3, R19 ;                                     /* 0x5c98078001370003 */
                                                                                     /* 0x001fc400f62007b1 */
        /*0048*/                   LDG.E R4, [R2] ;                                  /* 0xeed4200000070204 */
        /*0050*/                   LDG.E R5, [R2+0x4] ;                              /* 0xeed4200000470205 */
        /*0058*/                   LDG.E R7, [R2+0x8] ;                              /* 0xeed4200000870207 */
                                                                                     /* 0x001fc400f62007b1 */
        /*0068*/                   LDG.E R8, [R2+0xc] ;                              /* 0xeed4200000c70208 */
        /*0070*/                   LDG.E R9, [R2+0x10] ;                             /* 0xeed4200001070209 */
        /*0078*/                   LDG.E R11, [R2+0x14] ;                            /* 0xeed420000147020b */
                                                                                     /* 0x001fc400fe2007f1 */
        /*0088*/                   LDG.E R12, [R2+0x18] ;                            /* 0xeed420000187020c */
        /*0090*/                   LDG.E R13, [R2+0x1c] ;                            /* 0xeed4200001c7020d */
        /*0098*/                   LDG.E R19, [R2+0x20] ;                            /* 0xeed4200002070213 */
                                                                                     /* 0x001fc400fe2007f1 */
        /*00a8*/                   LDG.E R20, [R2+0x24] ;                            /* 0xeed4200002470214 */
        /*00b0*/                   LDG.E R21, [R2+0x28] ;                            /* 0xeed4200002870215 */
        /*00b8*/                   LDG.E R23, [R2+0x2c] ;                            /* 0xeed4200002c70217 */
                                                                                     /* 0x001ec400f62007f1 */
        /*00c8*/                   LDG.E R18, [R2+0x30] ;                            /* 0xeed4200003070212 */
        /*00d0*/                   LDG.E R17, [R2+0x34] ;                            /* 0xeed4200003470211 */
        /*00d8*/                   LDG.E R16, [R2+0x38] ;                            /* 0xeed4200003870210 */
                                                                                     /* 0x001ec400f62007b1 */
        /*00e8*/                   LDG.E R15, [R2+0x3c] ;                            /* 0xeed4200003c7020f */
        /*00f0*/                   LDG.E R6, [R2+0x40] ;                             /* 0xeed4200004070206 */
        /*00f8*/                   LDG.E R10, [R2+0x44] ;                            /* 0xeed420000447020a */
                                                                                     /* 0x001f8400fe6007b1 */
        /*0108*/                   LDG.E R14, [R2+0x48] ;                            /* 0xeed420000487020e */
        /*0110*/                   DEPBAR.LE SB5, 0x7 ;                              /* 0xf0f0000034770000 */
        /*0118*/                   XMAD R22, R4, c[0x0] [0x148], RZ ;                /* 0x4e007f8005270416 */
                                                                                     /* 0x001fc440fe2207f1 */
        /*0128*/                   XMAD.MRG R25, R4.reuse, c[0x0] [0x148].H1, RZ ;   /* 0x4f107f8005270419 */
        /*0130*/                   XMAD R24, R5.reuse, c[0x0] [0x148], RZ ;          /* 0x4e007f8005270518 */
        /*0138*/                   XMAD.MRG R26, R5, c[0x0] [0x148].H1, RZ ;         /* 0x4f107f800527051a */
                                                                                     /* 0x081fc000fe4007f3 */
        /*0148*/                   XMAD.MRG R27, R8, c[0x0] [0x148].H1, RZ ;         /* 0x4f107f800527081b */
        /*0150*/                   XMAD.PSL.CBCC R4, R4.H1, R25.H1, R22 ;            /* 0x5b300b1801970404 */
        /*0158*/         {         XMAD R22, R7.reuse, c[0x0] [0x148], RZ ;          /* 0x4e007f8005270716 */
                                                                                     /* 0x001fc400fc2007f1 */
        /*0168*/                   STG.E [R2], R4         }
                                                                                     /* 0xeedc200000070204 */
        /*0170*/                   XMAD.MRG R25, R7, c[0x0] [0x148].H1, RZ ;         /* 0x4f107f8005270719 */
        /*0178*/                   XMAD.PSL.CBCC R5, R5.H1, R26.H1, R24 ;            /* 0x5b300c1801a70505 */
                                                                                     /* 0x001fc400fe6007f0 */
        /*0188*/         {         XMAD R26, R8, c[0x0] [0x148], RZ ;                /* 0x4e007f800527081a */
        /*0190*/                   DEPBAR.LE SB5, 0x5         }
                                                                                     /* 0xf0f0000034570000 */
        /*0198*/                   STG.E [R2+0x4], R5 ;                              /* 0xeedc200000470205 */
                                                                                     /* 0x001f8800fe2007f1 */
        /*01a8*/                   XMAD.MRG R24, R9, c[0x0] [0x148].H1, RZ ;         /* 0x4f107f8005270918 */
        /*01b0*/                   XMAD.MRG R28, R17, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f800527111c */
        /*01b8*/                   XMAD.PSL.CBCC R7, R7.H1, R25.H1, R22 ;            /* 0x5b300b1801970707 */
                                                                                     /* 0x001fc800fe2207f0 */
        /*01c8*/         {         XMAD R22, R9.reuse, c[0x0] [0x148], RZ ;          /* 0x4e007f8005270916 */
        /*01d0*/                   STG.E [R2+0x8], R7         }
                                                                                     /* 0xeedc200000870207 */
        /*01d8*/                   XMAD.PSL.CBCC R8, R8.H1, R27.H1, R26 ;            /* 0x5b300d1801b70808 */
                                                                                     /* 0x001fc400fe2207f0 */
        /*01e8*/         {         XMAD R25, R11.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270b19 */
        /*01f0*/                   STG.E [R2+0xc], R8         }
                                                                                     /* 0xeedc200000c70208 */
        /*01f8*/                   XMAD.MRG R26, R11, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270b1a */
                                                                                     /* 0x081fc000fc4007f1 */
        /*0208*/                   XMAD.MRG R27, R12, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270c1b */
        /*0210*/                   XMAD.PSL.CBCC R9, R9.H1, R24.H1, R22 ;            /* 0x5b300b1801870909 */
        /*0218*/         {         XMAD R22, R12.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270c16 */
                                                                                     /* 0x001fc840fe2007f1 */
        /*0228*/                   STG.E [R2+0x10], R9         }
                                                                                     /* 0xeedc200001070209 */
        /*0230*/                   XMAD R24, R13.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270d18 */
        /*0238*/                   XMAD.PSL.CBCC R11, R11.H1, R26.H1, R25 ;          /* 0x5b300c9801a70b0b */
                                                                                     /* 0x001fc400fe2007f0 */
        /*0248*/         {         XMAD.MRG R26, R13, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270d1a */
        /*0250*/                   STG.E [R2+0x14], R11         }
                                                                                     /* 0xeedc20000147020b */
        /*0258*/                   XMAD.MRG R25, R19, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005271319 */
                                                                                     /* 0x001fc440fe0007e2 */
        /*0268*/                   XMAD.PSL.CBCC R12, R12.H1, R27.H1, R22 ;          /* 0x5b300b1801b70c0c */
        /*0270*/         {         XMAD R22, R19.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005271316 */
        /*0278*/                   STG.E [R2+0x18], R12         }
                                                                                     /* 0xeedc20000187020c */
                                                                                     /* 0x001fc000fe4207f1 */
        /*0288*/                   XMAD.MRG R27, R20.reuse, c[0x0] [0x148].H1, RZ ;  /* 0x4f107f800527141b */
        /*0290*/                   XMAD.PSL.CBCC R13, R13.H1, R26.H1, R24 ;          /* 0x5b300c1801a70d0d */
        /*0298*/         {         XMAD R26, R20, c[0x0] [0x148], RZ ;               /* 0x4e007f800527141a */
                                                                                     /* 0x001f8800fe2007f1 */
        /*02a8*/                   STG.E [R2+0x1c], R13         }
                                                                                     /* 0xeedc200001c7020d */
        /*02b0*/                   XMAD.MRG R24, R21, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005271518 */
        /*02b8*/                   XMAD.PSL.CBCC R19, R19.H1, R25.H1, R22 ;          /* 0x5b300b1801971313 */
                                                                                     /* 0x081fc400fe2207f0 */
        /*02c8*/         {         XMAD R22, R21.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005271516 */
        /*02d0*/                   STG.E [R2+0x20], R19         }
                                                                                     /* 0xeedc200002070213 */
        /*02d8*/                   XMAD R25, R23.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005271719 */
                                                                                     /* 0x001fc400fe0007f2 */
        /*02e8*/                   XMAD.PSL.CBCC R20, R20.H1, R27.H1, R26 ;          /* 0x5b300d1801b71414 */
        /*02f0*/         {         XMAD.MRG R26, R23, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f800527171a */
        /*02f8*/                   STG.E [R2+0x24], R20         }
                                                                                     /* 0xeedc200002470214 */
                                                                                     /* 0x081fc000fc4007f1 */
        /*0308*/                   XMAD.MRG R27, R18, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f800527121b */
        /*0310*/                   XMAD.PSL.CBCC R21, R21.H1, R24.H1, R22 ;          /* 0x5b300b1801871515 */
        /*0318*/         {         XMAD R24, R18.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005271218 */
                                                                                     /* 0x001fc000fe2007f1 */
        /*0328*/                   STG.E [R2+0x28], R21         }
                                                                                     /* 0xeedc200002870215 */
        /*0330*/                   XMAD R22, R17, c[0x0] [0x148], RZ ;               /* 0x4e007f8005271116 */
        /*0338*/         {         XMAD.PSL.CBCC R23, R23.H1, R26.H1, R25 ;          /* 0x5b300c9801a71717 */
                                                                                     /* 0x001fc000fe6007b1 */
        /*0348*/                   LDG.E R26, [R2+0x50]         }
                                                                                     /* 0xeed420000507021a */
        /*0350*/                   DEPBAR.LE SB5, 0x3 ;                              /* 0xf0f0000034370000 */
        /*0358*/         {         XMAD.MRG R25, R16, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005271019 */
                                                                                     /* 0x081fc000fc4007f1 */
        /*0368*/                   STG.E [R2+0x2c], R23         }
                                                                                     /* 0xeedc200002c70217 */
        /*0370*/                   XMAD.PSL.CBCC R24, R18.H1, R27.H1, R24 ;          /* 0x5b300c1801b71218 */
        /*0378*/         {         XMAD R18, R16.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005271012 */
                                                                                     /* 0x081fc000fe4007f1 */
        /*0388*/                   STG.E [R2+0x30], R24         }
                                                                                     /* 0xeedc200003070218 */
        /*0390*/                   XMAD.PSL.CBCC R17, R17.H1, R28.H1, R22 ;          /* 0x5b300b1801c71111 */
        /*0398*/         {         XMAD R22, R15.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270f16 */
                                                                                     /* 0x081fc400fe2007f1 */
        /*03a8*/                   STG.E [R2+0x34], R17         }
                                                                                     /* 0xeedc200003470211 */
        /*03b0*/                   XMAD.MRG R28, R15, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270f1c */
        /*03b8*/                   XMAD.MRG R27, R6.reuse, c[0x0] [0x148].H1, RZ ;   /* 0x4f107f800527061b */
                                                                                     /* 0x001fc000f62007f0 */
        /*03c8*/         {         XMAD.PSL.CBCC R16, R16.H1, R25.H1, R18 ;          /* 0x5b30091801971010 */
        /*03d0*/                   LDG.E R18, [R2+0x4c]         }
                                                                                     /* 0xeed4200004c70212 */
        /*03d8*/         {         XMAD R25, R6, c[0x0] [0x148], RZ ;                /* 0x4e007f8005270619 */
                                                                                     /* 0x001f8400fe2007f3 */
        /*03e8*/                   DEPBAR.LE SB5, 0x2         }
                                                                                     /* 0xf0f0000034270000 */
        /*03f0*/                   STG.E [R2+0x38], R16 ;                            /* 0xeedc200003870210 */
        /*03f8*/                   XMAD.MRG R29, R10, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270a1d */
                                                                                     /* 0x081fc000f64007f0 */
        /*0408*/         {         XMAD.PSL.CBCC R15, R15.H1, R28.H1, R22 ;          /* 0x5b300b1801c70f0f */
        /*0410*/                   LDG.E R22, [R2+0x54]         }
                                                                                     /* 0xeed4200005470216 */
        /*0418*/         {         XMAD R28, R10.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270a1c */
                                                                                     /* 0x001fc840fe2007f1 */
        /*0428*/                   STG.E [R2+0x3c], R15         }
                                                                                     /* 0xeedc200003c7020f */
        /*0430*/                   XMAD R31, R14.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270e1f */
        /*0438*/                   XMAD.PSL.CBCC R25, R6.H1, R27.H1, R25 ;           /* 0x5b300c9801b70619 */
                                                                                     /* 0x001fc000fe4207f0 */
        /*0448*/         {         XMAD.MRG R6, R14.reuse, c[0x0] [0x148].H1, RZ ;   /* 0x4f107f8005270e06 */
        /*0450*/                   STG.E [R2+0x40], R25         }
                                                                                     /* 0xeedc200004070219 */
        /*0458*/         {         XMAD.PSL.CBCC R27, R10.H1, R29.H1, R28 ;          /* 0x5b300e1801d70a1b */
                                                                                     /* 0x001ec400fe0007b4 */
        /*0468*/                   LDG.E R10, [R2+0x5c]         }
                                                                                     /* 0xeed4200005c7020a */
        /*0470*/         {         XMAD.PSL.CBCC R28, R14.H1, R6.H1, R31 ;           /* 0x5b300f9800670e1c */
        /*0478*/                   LDG.E R14, [R2+0x58]         }
                                                                                     /* 0xeed420000587020e */
                                                                                     /* 0x001fc000f7a007f0 */
        /*0488*/         {         IADD32I R0, R0, 0x19 ;                            /* 0x1c00000001970000 */
        /*0490*/                   LDG.E R6, [R2+0x60]         }
                                                                                     /* 0xeed4200006070206 */
        /*0498*/         {         ISETP.NE.AND P0, PT, R0, RZ, PT ;                 /* 0x5b6b03800ff70007 */
                                                                                     /* 0x001fcc001d8000f1 */
        /*04a8*/                   STG.E [R2+0x44], R27         }
                                                                                     /* 0xeedc20000447021b */
        /*04b0*/                   STG.E [R2+0x48], R28 ;                            /* 0xeedc20000487021c */
        /*04b8*/                   DEPBAR.LE SB5, 0x5 ;                              /* 0xf0f0000034570000 */
                                                                                     /* 0x001fcc00fe2007e1 */
        /*04c8*/                   XMAD R4, R26, c[0x0] [0x148], RZ ;                /* 0x4e007f8005271a04 */
        /*04d0*/                   XMAD.MRG R5, R26, c[0x0] [0x148].H1, RZ ;         /* 0x4f107f8005271a05 */
        /*04d8*/                   DEPBAR.LE SB5, 0x3 ;                              /* 0xf0f0000034370000 */
                                                                                     /* 0x081fc400fe2207f1 */
        /*04e8*/                   XMAD R29, R18.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f800527121d */
        /*04f0*/                   XMAD.MRG R31, R18, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f800527121f */
        /*04f8*/                   XMAD R7, R22.reuse, c[0x0] [0x148], RZ ;          /* 0x4e007f8005271607 */
                                                                                     /* 0x001fc000fe4007e4 */
        /*0508*/                   XMAD.MRG R8, R22, c[0x0] [0x148].H1, RZ ;         /* 0x4f107f8005271608 */
        /*0510*/                   XMAD.PSL.CBCC R29, R18.H1, R31.H1, R29 ;          /* 0x5b300e9801f7121d */
        /*0518*/         {         IADD32I R18.CC, R2, 0x64 ;                        /* 0x1c10000006470212 */
                                                                                     /* 0x001fc460fe2000ed */
        /*0528*/                   STG.E [R2+0x4c], R29         }
                                                                                     /* 0xeedc200004c7021d */
        /*0530*/                   XMAD R12, R10.reuse, c[0x0] [0x148], RZ ;         /* 0x4e007f8005270a0c */
        /*0538*/                   XMAD.MRG R13, R10, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270a0d */
                                                                                     /* 0x001f8400fe2207f1 */
        /*0548*/                   XMAD R9, R14.reuse, c[0x0] [0x148], RZ ;          /* 0x4e007f8005270e09 */
        /*0550*/                   XMAD.MRG R11, R14, c[0x0] [0x148].H1, RZ ;        /* 0x4f107f8005270e0b */
        /*0558*/                   XMAD R19, R6, c[0x0] [0x148], RZ ;                /* 0x4e007f8005270613 */
                                                                                     /* 0x001fc000fe4207f1 */
        /*0568*/                   XMAD.MRG R20, R6.reuse, c[0x0] [0x148].H1, RZ ;   /* 0x4f107f8005270614 */
        /*0570*/                   XMAD.PSL.CBCC R4, R26.H1, R5.H1, R4 ;             /* 0x5b30021800571a04 */
        /*0578*/         {         XMAD.PSL.CBCC R5, R22.H1, R8.H1, R7 ;             /* 0x5b30039800871605 */
                                                                                     /* 0x0003c800fe0000f2 */
        /*0588*/                   STG.E [R2+0x50], R4         }
                                                                                     /* 0xeedc200005070204 */
        /*0590*/         {         XMAD.PSL.CBCC R7, R14.H1, R11.H1, R9 ;            /* 0x5b30049800b70e07 */
        /*0598*/                   STG.E [R2+0x54], R5         }
                                                                                     /* 0xeedc200005470205 */
                                                                                     /* 0x001fc0001e4007f0 */
        /*05a8*/         {         XMAD.PSL.CBCC R8, R10.H1, R13.H1, R12 ;           /* 0x5b30061800d70a08 */
        /*05b0*/                   STG.E [R2+0x58], R7         }
                                                                                     /* 0xeedc200005870207 */
        /*05b8*/         {         XMAD.PSL.CBCC R9, R6.H1, R20.H1, R19 ;            /* 0x5b30099801470609 */
                                                                                     /* 0x0003c400fe0000f2 */
        /*05c8*/                   STG.E [R2+0x5c], R8         }
                                                                                     /* 0xeedc200005c70208 */
        /*05d0*/         {         IADD.X R19, RZ, R3 ;                              /* 0x5c1008000037ff13 */
        /*05d8*/                   STG.E [R2+0x60], R9         }
                                                                                     /* 0xeedc200006070209 */
                                                                                     /* 0x001ffc00ffe007fd */
        /*05e8*/               @P0 BRA 0x30 ;                                        /* 0xe2400fffa400000f */
        /*05f0*/                   EXIT ;                                            /* 0xe30000000007000f */
        /*05f8*/                   BRA 0x5f8 ;                                       /* 0xe2400fffff87000f */
		..........



Fatbin ptx code:
================
arch = sm_52
code version = [7,0]
producer = <unknown>
host = linux
compile_size = 64bit
compressed
