///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
This code was written by Benjamin Berczi as part of the PhD project titled "Simulating Semiclassical Black Holes" from the 
University of Nottingham.

We (Sahota, Pandita and Singh) modify it for the case of a test quantum field propagating on the background scalar field 
collapsing to form a black hole.

It is a self-contained C file that simulates a massless scalar field coupled to Einstein gravity in the ADM formulation 
with test quantum scalar field.

Details may be found in Benjamin Berczi's publications and PhD thesis.

*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// gcc Alcubierre_classsical_collapse_TQF_final.c -fopenmp -lgslcblas -lm  -lgsl -g ./a.out
// gcc Alcubierre_classsical_collapse_TQF_final.c -fopenmp -lgslcblas -lm  -lgsl -g ./a.out
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initialising the libraries for the code */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <complex.h>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initialising the constants and parameters */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* CONSANTS */
#define         PI                                 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define         M_P                                1.0                                                                               //sqrt(8*PI)
#define         c                                  1.0                                                                               // speed of light

/* GRID PARAMETERS */
#define         lattice_size                       500
#define         buff_size                          10
#define         lattice_size_buff                  510

#define         dr                                 0.025                                                                             // size of the spatial grid spacing
#define         dt                                 dr*0.25                                                                           // size of the temporal grid spacing

/* SCALAR FIELD PARAMETERS */
#define         amplitude                          5.0                                                                               // amplitude of the gaussian scalar field
#define         mass                               0.0
#define         initial_radius                     0.0                                                                               // initial radius of the gaussian scalar field
#define         initial_width                      1.0                                                                               // initial width of the gaussian scalar field

/* Quantum SCALAR FIELD PARAMETERS: expectation in coherent state */
#define         q_amplitude                          0.001                                                                            // amplitude of the gaussian scalar field
#define         q_mass                               0.0
#define         q_initial_radius                     0.0                                                                             // initial radius of the gaussian scalar field
#define         q_initial_width                      0.001

/* QUANTUM OR CLASSICAL SIMULATION */
#define         hbar                               1                                                                                 // set to 1 for quantum, 0 for classical. This just sets the backreaction, and is in set_bi_linears.c, the quantum modes are still evolved
#define         coherent_state_switch              1                                                                                 // set to 0 to just have the mode functions

/*Artificial dissipation parameters*/
#define         damping_order                      4                                                                                  // order of damping
#define         epsilonc                           0.1                                                                                // epsilon is the constant in the damping term, it's max value is 0.5
#define         epsilonq                           0.5                                                                                // epsilon is the constant in the damping term, it's max value is 0.5

/* QUANTUM GHOST FIELD PARAMETERS */
#define         number_of_q_fields                 6                                                                                 // number of quantum fields, 1 real, 5 ghosts for regularisation
#define         muSq                               0.0                                                                               // mass of scalar field
#define         mSqGhost                           2.0                                                                               // base mass of the Pauli-Villars regulator fields
// double          massSq[number_of_q_fields] = {muSq};                                                                              // masses of the ghost fields
double          massSq[number_of_q_fields] = { muSq, mSqGhost, 3.0 * mSqGhost, mSqGhost, 3.0 * mSqGhost, 4.0 * mSqGhost};            // masses of the ghost fields
// double          ghost_or_physical[number_of_q_fields] = { 1 };                                                                    // distinguishing between the real and ghost fields
double          ghost_or_physical[number_of_q_fields] = { 1, -1, 1, -1, 1, -1};                                                      // distinguishing between the real and ghost fields

/* QUANTUM MODE PARAMETERS */
#define         dk                                 1.0*PI/15.0          
#define         k_min                              1.0*PI/15.0                                                                       // minimum value of k, also =dk
#define         number_of_k_modes                  50                                                                                // number of k modes
#define         number_of_l_modes                  50                                                                                // number of l modes
#define         k_start                            0
#define         l_start                            0                                                                                 //the range of l is l_start, l_start+l_step, l_start+2l_step...
#define         l_step                             1

/* SIMULATION PARAMETERS */
#define         evolve_time                        1.0
#define         evolve_time_int                    (int)(1010)
#define         per_five                           (int)(1)
#define         evolve_time_int_per_five           (int)(1010)
#define         threads_number                    (int)(24)

/* NUMERICAL TIME ITERATION PARAMETERS */
#define         nu_legendre                        5                                                                                  // order of the Legendre gaussian quadrature
#define         number_of_RK_implicit_iterations   10                                                                                 // number of iterations for the time evolution

#define         ADM_loc                            960

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Defining structures */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// classical field or coherent state expectation value
struct classical_fields{
    double *pi;
    double *phi;
    double *chi;
    double *q_pi;
    double *q_phi;
    double *q_chi;
}; 
typedef struct classical_fields Classical_fields;
// quantum mode functions 
struct quantum_fields{
    __complex__ double ***pi;
    __complex__ double ***phi;
    __complex__ double ***chi;
}; 
typedef struct quantum_fields Quantum_fields;
// metric variables
struct metric_fields{
    double *A;
    double *B;
    double *D_B;
    double *U_tilda;
    double *K;
    double *K_B;
    double *lambda;
    double *alpha;
    double *D_alpha;
}; 
typedef struct metric_fields Metric_Fields;
// stress-energy tensor components for bg field
struct stress_tensor{
    double rho;
    double j_A;
    double S_A;
    double S_B;
    double q_rho;
    double q_j_A;
    double q_S_A;
    double q_S_B;
}; 
typedef struct stress_tensor Stress_Tensor;
// bilinears of the test quantum field
struct bi_linears{
    double phi_phi;
    double chi_chi;
    double pi_pi;
    double chi_pi;
    double del_theta_phi_del_theta_phi_over_r_sq;
};
typedef struct bi_linears Bi_Linears;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Making the points for the spatial grid */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void make_points(double r[lattice_size_buff]){
    for (int i=buff_size;i<lattice_size_buff;++i){
            r[i]=(i-buff_size)*dr;                                                   // uniform grid
    }
    for (int i=1;i<buff_size+1;++i){
            r[buff_size-i]=-i*dr;                                                    // uniform grid
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function to set the buff zone of the spatial grid for all fields */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_buff_zone(Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric_fields){
    #pragma omp parallel for num_threads(threads_number)
    for (int i=1; i<buff_size+1; i++){
        c_fields->phi[buff_size-i] = c_fields->phi[buff_size+i];
        c_fields->chi[buff_size-i] =-c_fields->chi[buff_size+i];
        c_fields->pi[buff_size-i]  = c_fields->pi[buff_size+i];

        c_fields->q_phi[buff_size-i] = c_fields->q_phi[buff_size+i];
        c_fields->q_chi[buff_size-i] =-c_fields->q_chi[buff_size+i];
        c_fields->q_pi[buff_size-i]  = c_fields->q_pi[buff_size+i];

        metric_fields->A[buff_size-i]      =  metric_fields->A[buff_size+i];
        metric_fields->B[buff_size-i]      =  metric_fields->B[buff_size+i];
        metric_fields->D_B[buff_size-i]    = -metric_fields->D_B[buff_size+i];
        metric_fields->U_tilda[buff_size-i]= -metric_fields->U_tilda[buff_size+i];
        metric_fields->K[buff_size-i]      =  metric_fields->K[buff_size+i];
        metric_fields->K_B[buff_size-i]    =  metric_fields->K_B[buff_size+i];
        metric_fields->lambda[buff_size-i] = -metric_fields->lambda[buff_size+i];
        metric_fields->alpha[buff_size-i]  =  metric_fields->alpha[buff_size+i];
        metric_fields->D_alpha[buff_size-i]= -metric_fields->D_alpha[buff_size+i];

        //for k and l values
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields[which_q_field]->phi[k][l][buff_size-i]   = q_fields[which_q_field]->phi[k][l][buff_size+i];
                    q_fields[which_q_field]->chi[k][l][buff_size-i]   =-q_fields[which_q_field]->chi[k][l][buff_size+i];
                    q_fields[which_q_field]->pi[k][l][buff_size-i]    = q_fields[which_q_field]->pi[k][l][buff_size+i];
                }
            }
        }
    }
    c_fields->chi[buff_size]           = 0.0;
    c_fields->q_chi[buff_size]           = 0.0;
    metric_fields->D_alpha[buff_size]  = 0.0;
    metric_fields->D_B[buff_size]      = 0.0;
    metric_fields->U_tilda[buff_size]  = 0.0;
    metric_fields->lambda[buff_size]   = 0.0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Calculating constants for the gaussian quadrature time iteration */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_c_i(double c_i[nu_legendre]){//these were calculated in Mathematics using, for example, N[Roots[LegendreP[6, x] == 0, x], 20]
    double zeros_of_P[nu_legendre];//={0.0};

    if(nu_legendre==2){
        zeros_of_P[0] = -sqrt(3.0)/3.0;
        zeros_of_P[1] =  sqrt(3.0)/3.0;
    }
    if(nu_legendre==3){
        zeros_of_P[0] = -sqrt(15.0)/5.0;
        zeros_of_P[1] =  0.0;
        zeros_of_P[2] =  sqrt(15.0)/5.0;
    }
    if(nu_legendre==4){
        zeros_of_P[0] = -sqrt(525.0+70.0*sqrt(30.0))/35.0;
        zeros_of_P[1] = -sqrt(525.0-70.0*sqrt(30.0))/35.0;
        zeros_of_P[2] =  sqrt(525.0-70.0*sqrt(30.0))/35.0;
        zeros_of_P[3] =  sqrt(525.0+70.0*sqrt(30.0))/35.0;
    }
    if(nu_legendre==5){
        zeros_of_P[0] = -sqrt(245.0+14.0*sqrt(70.0))/21.0;
        zeros_of_P[1] = -sqrt(245.0-14.0*sqrt(70.0))/21.0;
        zeros_of_P[2] =  0.0;
        zeros_of_P[3] =  sqrt(245.0-14.0*sqrt(70.0))/21.0;
        zeros_of_P[4] =  sqrt(245.0+14.0*sqrt(70.0))/21.0;
    }
    if(nu_legendre==6){
        zeros_of_P[0] = -0.93246951420315202781;
        zeros_of_P[1] = -0.66120938646626451366;
        zeros_of_P[2] = -0.23861918608319690863;
        zeros_of_P[3] =  0.23861918608319690863;
        zeros_of_P[4] =  0.66120938646626451366;
        zeros_of_P[5] =  0.93246951420315202781;
    }
    if(nu_legendre==7){
        zeros_of_P[0] = -0.94910791234275852453;
        zeros_of_P[1] = -0.74153118559939443986;
        zeros_of_P[2] = -0.40584515137739716691;
        zeros_of_P[3] =  0.0;
        zeros_of_P[4] =  0.40584515137739716691;
        zeros_of_P[5] =  0.74153118559939443986;
        zeros_of_P[6] =  0.94910791234275852453;
    }

    for(int i=0;i<nu_legendre;++i){
        c_i[i] = (zeros_of_P[i]+1.0)/2.0 ;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void find_a_ij__b_i(double c_i[nu_legendre], double b_i[nu_legendre], double a_ij[nu_legendre][nu_legendre], double GL_matrix_inverse[nu_legendre][nu_legendre]){
    double RHS_vector1[nu_legendre], RHS_vector2[nu_legendre];

    for(int row=0;row<nu_legendre;++row){
        for(int j=0;j<nu_legendre;++j){
            RHS_vector1[j] = pow(c_i[row],j+1)/(j+1);
            RHS_vector2[j] = 1.0/(j+1);
        }
        for(int i=0;i<nu_legendre;++i){
            a_ij[row][i]=0.0;
            for(int j=0;j<nu_legendre;++j){
                a_ij[row][i] = a_ij[row][i] + GL_matrix_inverse[i][j]*RHS_vector1[j];
            }
        }
    }

    for(int i=0;i<nu_legendre;++i){
        b_i[i] = 0.0;
        for(int j=0;j<nu_legendre;++j){
            b_i[i] = b_i[i] + GL_matrix_inverse[i][j]*RHS_vector2[j];
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-Gauss-Legendre matrix inverse---------------------
void find_GL_matrix_inverse(double c_i[nu_legendre], double GL_matrix_inverse[nu_legendre][nu_legendre]){
    double determinant, row_factor[nu_legendre], linear_sum[nu_legendre], quadratic_sum[nu_legendre], cubic_sum[nu_legendre], quartic_sum[nu_legendre];

    //first get the determinant
    determinant = 1.0;
    for(int i=0;i<nu_legendre;++i){
        for(int j=i+1;j<nu_legendre;++j){
            determinant = determinant*(c_i[j] - c_i[i]);
        }
    }

    //this gives determinants of  {(c1-c0); (c1-c0)(c2-c0)(c2-c1); (c1-c0)(c2-c0)(c3-c0)(c2-c1)(c3-c1)(c3-c2)}

    for(int row=0;row<nu_legendre;++row){
        row_factor[row]=1.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                if(i!=row && j!=row) row_factor[row] = row_factor[row]*(c_i[j]-c_i[i]);
            }
        }

        linear_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            if(i!=row)linear_sum[row] = linear_sum[row] + c_i[i];
        }

        quadratic_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                if(i!=row && j!=row)quadratic_sum[row] = quadratic_sum[row] + c_i[i]*c_i[j];
            }
        }
        cubic_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                for(int k=j+1;k<nu_legendre;++k){
                    if(i!=row && j!=row && k!=row)cubic_sum[row] = cubic_sum[row] + c_i[i]*c_i[j]*c_i[k];
                }
            }
        }
        quartic_sum[row] = 0.0;
        for(int i=0;i<nu_legendre;++i){
            for(int j=i+1;j<nu_legendre;++j){
                for(int k=j+1;k<nu_legendre;++k){
                    for(int l=k+1;l<nu_legendre;++l){
                        if(i!=row && j!=row && k!=row && l!=row)quartic_sum[row] = quartic_sum[row] + c_i[i]*c_i[j]*c_i[k]*c_i[l];
                    }
                }
            }
        }

    }
    for(int col=0;col<nu_legendre;++col){
        for(int row=0;row<nu_legendre;++row){
            if(col==0)GL_matrix_inverse[row][col] = row_factor[row]*quartic_sum[row]/determinant;
            if(col==1)GL_matrix_inverse[row][col] = row_factor[row]*cubic_sum[row]/determinant;
            if(col==2)GL_matrix_inverse[row][col] = row_factor[row]*quadratic_sum[row]/determinant;
            if(col==3)GL_matrix_inverse[row][col] = row_factor[row]*linear_sum[row]/determinant;
            if(col==4)GL_matrix_inverse[row][col] = row_factor[row]/determinant;
            GL_matrix_inverse[row][col] = pow(-1,(row+col))*GL_matrix_inverse[row][col];
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* First derivative function for real fields, uses 20 neighbouring points at the moment*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double first_deriv(int m, double *field){
    double deriv=0.0;

    if (m<lattice_size_buff-buff_size){

        //deriv=-(field[m+4]-32.0/3.0*field[m+3]+56.0*field[m+2]-224.0*field[m+1]+224.0*field[m-1]-56.0*field[m-2]+32.0/3.0*field[m-3]-field[m-4])/(280.0*dr);


        deriv = (6.097575365271568 * pow(10.0, 40) * field[m - 10] - 1.3550167478381302 * pow(10.0, 42) * field[m - 9] + 1.448174149252008 * pow(10.0, 43) * field[m - 8] - 9.930337023442402 * pow(10.0, 43) * field[m - 7]
                + 4.92379210745689 * pow(10.0, 44) * field[m - 6] - 1.8907361692634594 * pow(10.0, 45) * field[m - 5] + 5.90855052894837 * pow(10.0, 45) * field[m - 4] - 1.5756134743862517 * pow(10.0, 46) * field[m - 3]
                + 3.840557843816541 * pow(10.0, 46) * field[m - 2] - 1.024148758351083 * pow(10.0, 47) * field[m - 1] - 1.2569905698058102 * pow(10.0, 33) * field[m] + 1.024148758351106 * pow(10.0, 47) * field[m + 1]
                - 3.8405578438166756 * pow(10.0, 46) * field[m + 2] + 1.5756134743863493 * pow(10.0, 46) * field[m + 3] - 5.908550528948874 * pow(10.0, 45) * field[m + 4] + 1.8907361692636562 * pow(10.0, 45) * field[m + 5]
                - 4.923792107457477 * pow(10.0, 44) * field[m + 6] + 9.930337023443726 * pow(10.0, 43) * field[m + 7] - 1.448174149252219 * pow(10.0, 43) * field[m + 8] + 1.3550167478383397 * pow(10.0, 42) * field[m + 9]
                - 6.097575365272549 * pow(10.0, 40) * field[m + 10]) / (1.1265636341862039 * pow(10.0, 47) * dr);

    }
    else if (m != lattice_size_buff-1){
        deriv=(field[m+1]-field[m-1])/(2.0*dr);
         //deriv = (23100.0 * field[m - 12] - 302400.0 * field[m - 11] + 1829520.0 * field[m - 10] - 6776000.0 * field[m - 9] + 17151750.0 * field[m - 8] - 31363200.0 * field[m - 7] + 42688800.0 * field[m - 6] - 43908480.0 * field[m - 5] + 34303500.0 * field[m - 4] - 20328000.0 * field[m - 3] + 9147600.0 * field[m - 2] - 3326400.0 * field[m - 1] + 860210.0 * field[m]) / (27720.0 * dr);

    }
    else{
        deriv=(field[m]-field[m-1])/(dr);
    }

    return deriv;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* First derivative function for complex fields, uses 20 neighbouring points at the moment*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double first_deriv_comp(int m, __complex__ double *field){
    __complex__ double deriv=0.0;

    if (m<lattice_size_buff-buff_size){

        //deriv=-(field[m+4]-32.0/3.0*field[m+3]+56.0*field[m+2]-224.0*field[m+1]+224.0*field[m-1]-56.0*field[m-2]+32.0/3.0*field[m-3]-field[m-4])/(280.0*dr);


        deriv=(6.097575365271568*pow(10.0,40)*field[m-10]-1.3550167478381302*pow(10.0,42)*field[m-9]+1.448174149252008*pow(10.0,43)*field[m-8]-9.930337023442402*pow(10.0,43)*field[m-7]
        +4.92379210745689*pow(10.0,44)*field[m-6]-1.8907361692634594*pow(10.0,45)*field[m-5]+5.90855052894837*pow(10.0,45)*field[m-4]-1.5756134743862517*pow(10.0,46)*field[m-3]
        +3.840557843816541*pow(10.0,46)*field[m-2]-1.024148758351083*pow(10.0,47)*field[m-1]-1.2569905698058102*pow(10.0,33)*field[m]+1.024148758351106*pow(10.0,47)*field[m+1]
        -3.8405578438166756*pow(10.0,46)*field[m+2]+1.5756134743863493*pow(10.0,46)*field[m+3]-5.908550528948874*pow(10.0,45)*field[m+4]+1.8907361692636562*pow(10.0,45)*field[m+5]
        -4.923792107457477*pow(10.0,44)*field[m+6]+9.930337023443726*pow(10.0,43)*field[m+7]-1.448174149252219*pow(10.0,43)*field[m+8]+1.3550167478383397*pow(10.0,42)*field[m+9]
        -6.097575365272549*pow(10.0,40)*field[m+10])/(1.1265636341862039*pow(10.0,47)*dr);

    }
    else if (m != lattice_size_buff){
        deriv=(field[m+1]-field[m-1])/(2.0*dr);
        //deriv=(23100.0*field[m-12]-302400.0*field[m-11]+1829520.0*field[m-10]-6776000.0*field[m-9]+17151750.0*field[m-8]-31363200.0*field[m-7]+42688800.0*field[m-6]-43908480.0*field[m-5]+34303500.0*field[m-4]-20328000.0*field[m-3]+9147600.0*field[m-2]-3326400.0*field[m-1]+860210.0*field[m])/(27720.0*dr);
    }
    else{
        deriv=(field[m]-field[m-1])/(dr);
    }

    return deriv;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function for the artificial dissipation, aka fourth derivative function for real fields, uses 4 neighbouring points at the moment*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double fourth_deriv(int m, double *field){
    double der=0.0;

    if (m<lattice_size_buff-buff_size){
        //der=(field[m+2]-4.0*field[m+1]+6.0*field[m]-4.0*field[m-1]+field[m-2])/pow(dr, 4.0);
        
        der = (2.152198547886781*pow(10.0,133) * field[m - 10] - 5.30597476094236*pow(10.0,134) * field[m - 9] + 6.365997549976923*pow(10.0,135) * field[m - 8] - 4.973308793355959*pow(10.0,136) * field[m - 7] + 2.863057643183926*pow(10.0,137) * field[m - 6]
            - 1.3087023772291203*pow(10.0,138) * field[m - 5] + 5.035932719448925*pow(10.0,138) * field[m - 4] - 1.7320298147875656*pow(10.0,139) * field[m - 3] + 5.721367554653479*pow(10.0,139) * field[m - 2] - 1.2906606322972197*pow(10.0,140) * field[m - 1]
            + 1.7040604531116496*pow(10.0,140) * field[m + 0] - 1.290660405149481*pow(10.0,140) * field[m + 1] + 5.721366313699189*pow(10.0,139) * field[m + 2] - 1.732029331201294*pow(10.0,139) * field[m + 3] + 5.035930883803483*pow(10.0,138) * field[m + 4]
            - 1.30870177022071*pow(10.0,138) * field[m + 5] + 2.8630560175418586*pow(10.0,137) * field[m + 6] - 4.973305446769496*pow(10.0,136) * field[m + 7] + 6.365992596959947*pow(10.0,135) * field[m + 8] - 5.305970076416748*pow(10.0,134) * field[m + 9]
            + 2.1521964229110124*pow(10.0,133) * field[m + 10]) / (1.0760047710287274*pow(10.0,139) * pow(dr, 4));
    
    }
    else{
        der=0.0;
        //der=(field[m]-4.0*field[m-1]+6.0*field[m-2]-4.0*field[m-3]+field[m-4])/pow(dr, 4.0);
        
    }
    return der;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function for the artificial dissipation, twelfth derivative function for real fields*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double twelfth_deriv(int m, double* field) {
    double der = 0.0;

    if (m < lattice_size_buff - buff_size) {
      
        der = (field[m-6]-12.0*field[m-5]+66.0*field[m-4]-220.0*field[m-3]+495.0*field[m-2]-792.0*field[m-1]+924.0*field[m]-792.0*field[m+1]+495.0*field[m+2]-220.0*field[m+3]+66.0*field[m+4]-12.0*field[m+5]+field[m+6])/pow(dr, 12);
    }
    else {
        der = 0.0;
    }
    return der;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function for the artificial dissipation, aka fourth derivative function for complex fields, uses 4 neighbouring points at the moment*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double fourth_deriv_comp(int m, __complex__ double *field){
    __complex__ double der=0.0;

    if (m<lattice_size_buff-2){
         der=(field[m+2]-4.0*field[m+1]+6.0*field[m]-4.0*field[m-1]+field[m-2])/pow(dr, 4.0);
         
  
        // der = (2.152198547886781 * pow(10.0, 133) * field[m - 10] - 5.30597476094236 * pow(10.0, 134) * field[m - 9] + 6.365997549976923 * pow(10.0, 135) * field[m - 8] - 4.973308793355959 * pow(10.0, 136) * field[m - 7] + 2.863057643183926 * pow(10.0, 137) * field[m - 6]
        //     - 1.3087023772291203 * pow(10.0, 138) * field[m - 5] + 5.035932719448925 * pow(10.0, 138) * field[m - 4] - 1.7320298147875656 * pow(10.0, 139) * field[m - 3] + 5.721367554653479 * pow(10.0, 139) * field[m - 2] - 1.2906606322972197 * pow(10.0, 140) * field[m - 1]
        //     + 1.7040604531116496 * pow(10.0, 140) * field[m + 0] - 1.290660405149481 * pow(10.0, 140) * field[m + 1] + 5.721366313699189 * pow(10.0, 139) * field[m + 2] - 1.732029331201294 * pow(10.0, 139) * field[m + 3] + 5.035930883803483 * pow(10.0, 138) * field[m + 4]
        //     - 1.30870177022071 * pow(10.0, 138) * field[m + 5] + 2.8630560175418586 * pow(10.0, 137) * field[m + 6] - 4.973305446769496 * pow(10.0, 136) * field[m + 7] + 6.365992596959947 * pow(10.0, 135) * field[m + 8] - 5.305970076416748 * pow(10.0, 134) * field[m + 9]
        //     + 2.1521964229110124 * pow(10.0, 133) * field[m + 10]) / (1.0760047710287274 * pow(10.0, 139) * pow(dr, 4));

        
    }
    else{
        der=0.0;
        //der=(field[m]-4.0*field[m-1]+6.0*field[m-2]-4.0*field[m-3]+field[m-4])/pow(dr, 4.0);
        
    }
    return der;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function for the artificial dissipation, twelfth derivative function for complex fields*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__complex__ double twelfth_deriv_comp(int m, __complex__ double* field) {
    __complex__ double der = 0.0;

    if (m < lattice_size_buff - buff_size) {
      
        der = (field[m-6]-12.0*field[m-5]+66.0*field[m-4]-220.0*field[m-3]+495.0*field[m-2]-792.0*field[m-1]+924.0*field[m]-792.0*field[m+1]+495.0*field[m+2]-220.0*field[m+3]+66.0*field[m+4]-12.0*field[m+5]+field[m+6])/pow(dr, 12);
    }
    else {
        der = 0.0;
    }
    return der;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* This function provides a version of gsl's Bessel function that ignores any underflow error */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double gsl_sf_bessel_jl_safe(int l, double x){
    gsl_sf_result answer;
    gsl_error_handler_t *old_error_handler=gsl_set_error_handler_off ();    // turn off the error handler
    int error_code=gsl_sf_bessel_jl_e(l, x, &answer);                       //compute the answer, and construct an error code
    gsl_set_error_handler(old_error_handler); //reset the error handler
    if(error_code==GSL_SUCCESS){                                                  //if there's no error then return the correct answer
        return answer.val;
    }
    else{
        //printf ("error in gsl_sf_bessel_jl_safe: %s\n", gsl_strerror (error_code));
        //exit(1);
        if(error_code==GSL_EUNDRFLW){
            return 0.0;
        }
        else{
            printf ("error in gsl_sf_bessel_jl_safe: %s\n", gsl_strerror (error_code));
            exit(1);
        }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* These functions provides the initial profile functions */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double phi_mode_profile_0(double k, int l, double r){
    return (  sqrt(k/PI)*gsl_sf_bessel_jl_safe(l, k*r)/pow(r,l) );
}
//---
double phi_mode_profile_0_prime(double k, int l, double r){
    return ( -k*sqrt(k/PI)*gsl_sf_bessel_jl_safe(l+1, k*r)/pow(r,l) );
}
//---
double phi_mode_profile_massive(double msq, double k, int l, double r){
    return ( k/sqrt(PI*sqrt(k*k+msq))*gsl_sf_bessel_jl_safe (l, k*r)/pow(r,l) );
}
//---
double phi_mode_profile_massive_prime(double msq, double k, int l, double r){
    return ( -k*k/sqrt(PI*sqrt(k*k+msq))*gsl_sf_bessel_jl_safe (l+1, k*r)/pow(r,l) );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises the classical variables */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initial_conditions_classical(Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double rpoints[lattice_size_buff], r;
    make_points(rpoints);

    /* METRIC FIELDS */
    for (int i=0;i<lattice_size_buff;++i){
        metric->A[i]       = 1.0;
        metric->B[i]       = 1.0;
        metric->D_B[i]     = 0.0;
        metric->K[i]       = 0.0;
        metric->K_B[i]     = 0.0;
        metric->alpha[i]   = 1.0;
        metric->D_alpha[i] = 0.0;
        metric->lambda[i]  = 0.0;
        metric->U_tilda[i] = 0.0;
    }

    /* CLASSICAL background MATTER FIELDS */
    /* PHI */
    for (int i=buff_size;i<lattice_size_buff;++i){
        r=rpoints[i];
        // c_fields->phi[i] = amplitude * (exp(- 8.0 * r * r ) + exp(- 2.0 * (r-3.0) * (r-3.0) ));
        c_fields->phi[i] = amplitude * exp(- r * r / (initial_width * initial_width));
        // c_fields->phi[i] = amplitude * exp(- r * r / (2.0*initial_width * initial_width));
        // c_fields->phi[i]=(amplitude*(r/initial_width)*(r/initial_width)*exp(-(r-initial_radius)*(r-initial_radius)/(initial_width*initial_width)))+
        //                 (amplitude*(r/initial_width)*(r/initial_width)*exp(-(-r-initial_radius)*(-r-initial_radius)/(initial_width*initial_width)));
    }
    /* CHI */
    for (int i=buff_size; i<lattice_size_buff; ++i){
        r=rpoints[i];
        if (i>buff_size){
            // c_fields->chi[i] = -2.0 * amplitude * (8.0 * r * exp(- 8.0 * r * r ) + 2.0 * (r-3.0) * exp(- 2.0 * (r-3.0) * (r-3.0) ));
            c_fields->chi[i] = -2.0 * amplitude *r/ (initial_width * initial_width) * exp(-r * r / (initial_width * initial_width));
            // c_fields->chi[i]=(1/initial_width*(2.0*amplitude*(r/initial_width)*(1.0 - (r)*(r-initial_radius)/(initial_width*initial_width))
            //                     *exp(-(r-initial_radius)*(r-initial_radius)/(initial_width*initial_width))))+
            //                 (1/initial_width*(2.0*amplitude*(r/initial_width)*(1.0 + (r)*(-r-initial_radius)/(initial_width*initial_width))
            //                     *exp(-(-r-initial_radius)*(-r-initial_radius)/(initial_width*initial_width))));
        }
        else{
            c_fields->chi[i]=0.0;
        }
    }
    /* PI */

    for (int i=buff_size;i<lattice_size_buff;++i){
        c_fields->pi[i]=0;
    }

    /* test QUANTUM FIELD EXPECTATION */
    /* PHI */
    for (int i=buff_size;i<lattice_size_buff;++i){
        r=rpoints[i];
        c_fields->q_phi[i] = q_amplitude * exp(- r * r / (2.0*q_initial_width * q_initial_width));
        // c_fields->q_phi[i]=(q_amplitude*(r/q_initial_width)*(r/q_initial_width)*exp(-(r-q_initial_radius)*(r-q_initial_radius)/(q_initial_width*q_initial_width)))+
        //                 (q_amplitude*(r/q_initial_width)*(r/q_initial_width)*exp(-(-r-q_initial_radius)*(-r-q_initial_radius)/(q_initial_width*q_initial_width)));
    }
    /* CHI */
    for (int i=buff_size; i<lattice_size_buff; ++i){
        r=rpoints[i];
        if (i>buff_size){
            c_fields->q_chi[i] = - q_amplitude *r/ (q_initial_width * q_initial_width) * exp(-r * r / (2.0*q_initial_width * q_initial_width));
            // c_fields->q_chi[i]=(1/q_initial_width*(2.0*q_amplitude*(r/q_initial_width)*(1.0 - (r)*(r-q_initial_radius)/(q_initial_width*q_initial_width))
            //                     *exp(-(r-q_initial_radius)*(r-q_initial_radius)/(q_initial_width*q_initial_width))))+
            //                 (1/q_initial_width*(2.0*q_amplitude*(r/q_initial_width)*(1.0 + (r)*(-r-q_initial_radius)/(q_initial_width*q_initial_width))
            //                     *exp(-(-r-q_initial_radius)*(-r-q_initial_radius)/(q_initial_width*q_initial_width))));
        }
        else{
            c_fields->q_chi[i]=0.0;
        }
    }
    /* PI */

    for (int i=buff_size;i<lattice_size_buff;++i){
        c_fields->q_pi[i]=0;
    }
    
    
    set_buff_zone(c_fields, q_fields, metric);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that initialises the quantum variables */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initial_conditions_quantum(Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){

    /* QUANTUM MATTER FIELDS */
   
    //the initial data for the quantum vacuum modes phi
    for(int i=buff_size; i<lattice_size_buff; ++i){

        double k_wavenumber, omega_phi;
        int l_value;
        double r;
        r =(i-buff_size)*dr;
        for(int k=0; k<number_of_k_modes; ++k){
            k_wavenumber = (k_start+(k+1))*k_min;
            for(int l=0; l<number_of_l_modes; ++l){
                l_value = l_start + l*l_step;
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){//cycle through the quantum fields and initialize them
                    omega_phi = sqrt(k_wavenumber*k_wavenumber + massSq[which_q_field]);
                    if(massSq[which_q_field]==0){
                        if(i>buff_size){
                            q_fields[which_q_field]->phi[k][l][i]  = phi_mode_profile_0(k_wavenumber,l_value,r);                 //set the r!=0 zero values
                        }
                        else{
                            if(2*l_value+1<GSL_SF_DOUBLEFACT_NMAX){       //check that 1/gsl_sf_doublefact isn't too small
                                q_fields[which_q_field]->phi[k][l][i]  = sqrt(k_wavenumber/PI)*pow(k_wavenumber,l_value)/gsl_sf_doublefact(2*l_value+1);
                            }
                            else{
                                q_fields[which_q_field]->phi[k][l][i]  = 0.0;
                            }
                        }
                    }
                    else{
                        if(i>buff_size){
                            q_fields[which_q_field]->phi[k][l][i]  = phi_mode_profile_massive(massSq[which_q_field],k_wavenumber,l_value,r);
                        }
                        else{                                                                                                       //this is the value at the origin
                            if(2*l_value+1<GSL_SF_DOUBLEFACT_NMAX){//check that 1/gsl_sf_doublefact isn't too small
                                q_fields[which_q_field]->phi[k][l][i]  = k_wavenumber/sqrt(PI*omega_phi)*pow(k_wavenumber,l_value)/gsl_sf_doublefact(2*l_value+1);
                            }
                            else{
                                q_fields[which_q_field]->phi[k][l][i]  = 0.0;
                            }
                        }
                    }

                    //then sort out the momenta
                    q_fields[which_q_field]->pi[k][l][i]          = -I*omega_phi*q_fields[which_q_field]->phi[k][l][i];                 //note that this is a specification of pi, and not phi_dot
                }

            }
        }
    }



    //the initial data for the quantum vacuum modes chi
    for(int i=buff_size; i<lattice_size_buff; ++i){
        double r;
        r = (i - buff_size) * dr;
        double k_wavenumber, omega_phi;
        int l_value;
        for(int k=0; k<number_of_k_modes; ++k){
            k_wavenumber = (k_start + (k + 1))*k_min;
            for(int l=0; l<number_of_l_modes; ++l){
                l_value = l_start + l*l_step;
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){//cycle through the quantum fields and initialize their gradients
                    omega_phi = sqrt(k_wavenumber*k_wavenumber + massSq[which_q_field]);
                    if(massSq[which_q_field]==0){//the massless case
                        if(i>buff_size){
                            q_fields[which_q_field]->chi[k][l][i] =  phi_mode_profile_0_prime(k_wavenumber,l_value,r); //this is a place-holder for phi_prime, it will get replaced with psi
                        }
                        else{//this is the value at the origin
                            if(2*l_value+3<GSL_SF_DOUBLEFACT_NMAX){//check that 1/gsl_sf_doublefact isn't too small
                                q_fields[which_q_field]->chi[k][l][i] = -sqrt(k_wavenumber/PI)*pow(k_wavenumber,l_value+2)*r/gsl_sf_doublefact(2*l_value+3);
                            }
                            else{
                                q_fields[which_q_field]->chi[k][l][i] = 0.0;
                            }
                        }
                    }

                    else{//the massive case
                        if(i>buff_size){
                            q_fields[which_q_field]->chi[k][l][i] = phi_mode_profile_massive_prime(massSq[which_q_field],k_wavenumber,l_value,r); //this is a place-holder for phi_prime, it will get replaced with psi
                                                                                                                                                // in set_A_and_K_and_lapse_initial(...)
                        }
                        else{//this is the value at the origin
                            if(2*l_value+3<GSL_SF_DOUBLEFACT_NMAX){//check that 1/gsl_sf_doublefact isn't too small
                                q_fields[which_q_field]->chi[k][l][i] = -k_wavenumber*k_wavenumber/sqrt(PI*omega_phi)*pow(k_wavenumber,l_value+1)*r/gsl_sf_doublefact(2*l_value+3);
                            }
                            else{
                                q_fields[which_q_field]->chi[k][l][i] = 0.0;
                            }
                        }
                    }
                }

            }
        }
    }
    set_buff_zone(c_fields, q_fields, metric);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the norm */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double norm(__complex__ double number){
    double nor=0.0;
    nor=(pow((__real__ number),2.0)+pow((__imag__ number),2.0));
    return nor;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the bilinears background field */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears(int i, Bi_Linears *bi_linears, Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double r, r_l;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    __complex__ double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    int l_value;

    r = dr*(i-buff_size);

    phi_phi = 0.0;
    chi_chi = 0.0;
    pi_pi   = 0.0;
    chi_pi  = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    phi_phi = c_fields->phi[i]*c_fields->phi[i];
    chi_chi = c_fields->chi[i]*c_fields->chi[i];
    pi_pi   = c_fields->pi[i] *c_fields->pi[i];
    chi_pi  = c_fields->chi[i]*c_fields->pi[i];
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    
    bi_linears->phi_phi                                             = phi_phi;
    bi_linears->chi_chi                                             = chi_chi;
    bi_linears->pi_pi                                               = pi_pi;
    bi_linears->chi_pi                                              = chi_pi;
    bi_linears->del_theta_phi_del_theta_phi_over_r_sq               = del_theta_phi_del_theta_phi_over_r_sq;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the biliears in the midpoints of the iteration */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_bi_linears_midpoint(int i , Bi_Linears *bi_binears_midpoint, Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double r, r_l;
    double phi_phi=0.0, psi_psi=0.0, pi_pi=0.0, psi_pi=0.0, del_theta_phi_del_theta_phi_over_r_sq=0.0;
    __complex__ double Phi_mode, Psi_mode, Pi_mode;
    int l_value;

    r = dr*(i-buff_size) + 0.5*dr;
    phi_phi = 0.0;
    psi_psi = 0.0;
    pi_pi   = 0.0;
    psi_pi  = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    phi_phi = 0.25*(c_fields->phi[i] + c_fields->phi[i+1])*(c_fields->phi[i] + c_fields->phi[i+1]);
    psi_psi = 0.25*(c_fields->chi[i] + c_fields->chi[i+1])*(c_fields->chi[i] + c_fields->chi[i+1]);
    pi_pi   = 0.25*(c_fields->pi[i]  + c_fields->pi[i+1] )*(c_fields->pi[i]  + c_fields->pi[i+1] );
    psi_pi  = 0.25*(c_fields->chi[i] + c_fields->chi[i+1])*(c_fields->pi[i]  + c_fields->pi[i+1] );
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    bi_binears_midpoint->phi_phi                                             = phi_phi;
    bi_binears_midpoint->chi_chi                                             = psi_psi;
    bi_binears_midpoint->pi_pi                                               = pi_pi;
    bi_binears_midpoint->chi_pi                                              = psi_pi;
    bi_binears_midpoint->del_theta_phi_del_theta_phi_over_r_sq               = del_theta_phi_del_theta_phi_over_r_sq;
}

/* Bilinears for test field */

void set_bi_linears_q(int i, Bi_Linears *bi_linears, Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double r, r_l;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    __complex__ double Phi_mode, Phi_mode_plus, Chi_mode, Pi_mode;
    int l_value;

    r = dr*(i-buff_size);

    phi_phi = 0.0;
    chi_chi = 0.0;
    pi_pi   = 0.0;
    chi_pi  = 0.0;
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    if(coherent_state_switch!=0){
        phi_phi = c_fields->q_phi[i]*c_fields->q_phi[i];
        chi_chi = c_fields->q_chi[i]*c_fields->q_chi[i];
        pi_pi   = c_fields->q_pi[i] *c_fields->q_pi[i];
        chi_pi  = c_fields->q_chi[i]*c_fields->q_pi[i];
        del_theta_phi_del_theta_phi_over_r_sq = 0.0;
    }

    //note that these modes are actually modes of phi, where Phi = r^l phi
    //Phi = r^l phi
    //Pi  = r^l pi
    //Psi = lr^{l-1} u + r^l psi
    if(hbar!= 0){
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
                    l_value = l_start + l*l_step;
                    r_l = pow(r,l_value);


                    /* PHI MODE */
                    Phi_mode        = r_l*(q_fields[which_q_field]->phi[k][l][i]);

                    if(i==buff_size){
                        Phi_mode_plus   = pow(r+dr,l_value)*(q_fields[which_q_field]->phi[k][l][i+1]);
                    }

                    /* CHI MODE */
                    if(l_value==0){
                        Chi_mode = q_fields[which_q_field]->chi[k][l][i];
                    }
                    else if (l_value==1){
                        Chi_mode = q_fields[which_q_field]->phi[k][l][i]+r*q_fields[which_q_field]->chi[k][l][i];
                    }
                    else{
                        Chi_mode = l_value*pow(r,l_value-1)*q_fields[which_q_field]->phi[k][l][i]+r_l*(q_fields[which_q_field]->chi[k][l][i]);
                    }

                    /* PI MODE */

                    Pi_mode  = r_l*q_fields[which_q_field]->pi[k][l][i];


                    /* ACTUAL BILINEARS */
                    phi_phi = phi_phi  + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*norm(Phi_mode); // instead of norm
                    chi_chi = chi_chi  + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*norm(Chi_mode);
                    pi_pi   = pi_pi    + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*norm(Pi_mode);
                    chi_pi  = chi_pi   + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)* (__real__ (Pi_mode * conj(Chi_mode)));

                    if(i!=buff_size){
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*0.5*l_value*(l_value+1.0)*(2.0*l_value+1.0)*norm(Phi_mode)/(r*r);
                    }
                    else{//use the data at r=dr to estimate the r=0 case. This is only relevant for l=1
                        del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*0.5*l_value*(l_value+1.0)*(2.0*l_value+1.0)*norm(Phi_mode_plus)/(dr*dr);
                    }


                }
            }
        }
    }
    //printf("\n %.100f, ", norm(chi_mode));
    bi_linears->phi_phi                                             = phi_phi;
    bi_linears->chi_chi                                             = chi_chi;
    bi_linears->pi_pi                                               = pi_pi;
    bi_linears->chi_pi                                              = chi_pi;
    bi_linears->del_theta_phi_del_theta_phi_over_r_sq               = del_theta_phi_del_theta_phi_over_r_sq;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that calculates the biliears in the midpoints of the iteration */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void set_bi_linears_midpoint_q(int i , Bi_Linears *bi_binears_midpoint, Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
//     double r, r_l;
//     double phi_phi=0.0, psi_psi=0.0, pi_pi=0.0, psi_pi=0.0, del_theta_phi_del_theta_phi_over_r_sq=0.0;
//     __complex__ double Phi_mode, Psi_mode, Pi_mode;
//     int l_value;

//     r = dr*(i-buff_size) + 0.5*dr;
//     phi_phi = 0.0;
//     psi_psi = 0.0;
//     pi_pi   = 0.0;
//     psi_pi  = 0.0;
//     del_theta_phi_del_theta_phi_over_r_sq = 0.0;

//     if(coherent_state_switch!=0){
//         phi_phi = 0.25*(c_fields->q_phi[i] + c_fields->q_phi[i+1])*(c_fields->q_phi[i] + c_fields->q_phi[i+1]);
//         psi_psi = 0.25*(c_fields->q_chi[i] + c_fields->q_chi[i+1])*(c_fields->q_chi[i] + c_fields->q_chi[i+1]);
//         pi_pi   = 0.25*(c_fields->q_pi[i]  + c_fields->q_pi[i+1] )*(c_fields->q_pi[i]  + c_fields->q_pi[i+1] );
//         psi_pi  = 0.25*(c_fields->q_chi[i] + c_fields->q_chi[i+1])*(c_fields->q_pi[i]  + c_fields->q_pi[i+1] );
//         del_theta_phi_del_theta_phi_over_r_sq = 0.0;
//     }

//     //note that these modes are actually modes of phi, where Phi = r^l phi
//     //Phi = r^l phi
//     //Pi  = r^l pi
//     //Psi = lr^{l-1} u + r^l psi
//     if(hbar!= 0){
//         for(int k=0; k<number_of_k_modes; ++k){
//             for(int l=0; l<number_of_l_modes; ++l){
//                 for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
//                     l_value = l_start + l*l_step;
//                     r_l = pow(r,l_value);

//                     Phi_mode = 0.5*r_l*(q_fields[which_q_field]->phi[k][l][i] + q_fields[which_q_field]->phi[k][l][i+1]);

//                     if(l_value==0){
//                         Psi_mode = 0.5*(q_fields[which_q_field]->chi[k][l][i] + q_fields[which_q_field]->chi[k][l][i+1]);
//                     }
//                     else if (l_value==1){
//                         Psi_mode = 0.5*(q_fields[which_q_field]->phi[k][l][i] + q_fields[which_q_field]->phi[k][l][i+1])
//                                   +0.5*r*(q_fields[which_q_field]->chi[k][l][i] + q_fields[which_q_field]->chi[k][l][i+1]);
//                     }
//                     else{
//                         Psi_mode = 0.5*l_value*pow(r, l_value-1)*((q_fields[which_q_field]->phi[k][l][i]) + (q_fields[which_q_field]->phi[k][l][i+1]))
//                                   +0.5*r_l*(q_fields[which_q_field]->chi[k][l][i] + q_fields[which_q_field]->chi[k][l][i+1]);
//                     }

//                     Pi_mode  = 0.5*r_l*(q_fields[which_q_field]->pi[k][l][i]+ q_fields[which_q_field]->pi[k][l][i+1]);



//                     phi_phi = phi_phi   + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*norm(Phi_mode);
//                     psi_psi = psi_psi   + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*norm(Psi_mode);
//                     pi_pi   = pi_pi     + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*norm(Pi_mode);
//                     psi_pi  = psi_pi    + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*(2.0*l_value+1.0)*(__real__ (Pi_mode * conj(Psi_mode)));
//                     del_theta_phi_del_theta_phi_over_r_sq = del_theta_phi_del_theta_phi_over_r_sq + hbar*ghost_or_physical[which_q_field]*dk/(4.0*PI)*0.5*l_value*(l_value+1.0)*(2.0*l_value+1.0)*norm(Phi_mode)/(r*r);

//                 }
//             }
//         }
//     }

//     bi_binears_midpoint->phi_phi                                             = phi_phi;
//     bi_binears_midpoint->chi_chi                                             = psi_psi;
//     bi_binears_midpoint->pi_pi                                               = pi_pi;
//     bi_binears_midpoint->chi_pi                                              = psi_pi;
//     bi_binears_midpoint->del_theta_phi_del_theta_phi_over_r_sq               = del_theta_phi_del_theta_phi_over_r_sq;
// }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Setting the cosmological constant */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double set_cosm_constant(Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double q_rho, q_S_A, A, B;
    int i;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
    Bi_Linears    bi_linears;

    A   = 1.0;
    B   = 1.0;

    i = buff_size;


    set_bi_linears_q(i, &bi_linears, c_fields, q_fields, metric);

    phi_phi = bi_linears.phi_phi;
    chi_chi = bi_linears.chi_chi;
    pi_pi   = bi_linears.pi_pi;
    chi_pi  = bi_linears.chi_pi;
    del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

    q_rho = 1.0/(2.0*A)*(pi_pi/(B*B)+chi_chi)+1.0/B*del_theta_phi_del_theta_phi_over_r_sq;
    q_S_A = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / (B)*del_theta_phi_del_theta_phi_over_r_sq;
 

    return q_rho;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Setting the stress tensor components for given variable fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_stress_tensor(int i, Stress_Tensor *stress_tnsr, Bi_Linears bi_linears, Metric_Fields *metric, Classical_fields *c_fields, Quantum_fields **q_fields){
    
    double rho=0.0, j_A=0.0, S_A=0.0, S_B=0.0;
    double A, B;
    double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;

    rho = stress_tnsr->rho;
    j_A = stress_tnsr->j_A;
    S_A = stress_tnsr->S_A;
    S_B = stress_tnsr->S_B;

    A   = metric->A[i];
    B   = metric->B[i];


    phi_phi = c_fields->phi[i]*c_fields->phi[i];
    chi_chi = c_fields->chi[i]*c_fields->chi[i];
    pi_pi   = c_fields->pi[i] *c_fields->pi[i];
    chi_pi  = c_fields->chi[i]*c_fields->pi[i];
    del_theta_phi_del_theta_phi_over_r_sq = 0.0;

    rho = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / (B)*del_theta_phi_del_theta_phi_over_r_sq;
    j_A = -chi_pi / (sqrt(A) * B);
    S_A = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / (B)*del_theta_phi_del_theta_phi_over_r_sq;
    S_B = 1.0 / (2.0 * A) * (pi_pi / (B * B) - chi_chi);

    stress_tnsr->rho = rho;
    stress_tnsr->j_A = j_A;
    stress_tnsr->S_A = S_A;
    stress_tnsr->S_B = S_B;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Finding the initial dA_dr - note that only "classical" contribution present, since the quantum ones are cancelled out exactly by the cosmological constant */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double set_dA_dr_initial(double r, double A, Bi_Linears bi_linears){
    double d_A_dr;
    double chi;
    double B=1.0;

    // chi = -32.0 * amplitude * (r * exp(- 16.0 * r * r ) + 2.0 * (r-3.0) * exp(- 32.0 * (r-3.0) * (r-3.0) ));
    
    // chi = -2.0 * amplitude * (8.0 * r * exp(- 8.0 * r * r ) + 2.0 * (r-3.0) * exp(- 2.0 * (r-3.0) * (r-3.0) ));
    chi = -2.0 * amplitude *r/ (initial_width * initial_width) * exp(-r * r / (initial_width * initial_width));
    // chi = (1/initial_width*(2.0*amplitude*(r/initial_width)*(1.0 - (r)*(r-initial_radius)/(initial_width*initial_width))
    //                     *exp(-(r-initial_radius)*(r-initial_radius)/(initial_width*initial_width))))+
    //                 (1/initial_width*(2.0*amplitude*(r/initial_width)*(1.0 + (r)*(-r-initial_radius)/(initial_width*initial_width))
    //                     *exp(-(-r-initial_radius)*(-r-initial_radius)/(initial_width*initial_width))));

    d_A_dr =    A*(( r!=0.0 ? 1.0/r*(1.0-A) : 0.0)+r/(2.0*(M_P*M_P))*chi*chi);
    
    return d_A_dr;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Setting the initial A, U_tilde, and lambda */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void set_A_U_lambda_initial(Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    

    metric->A[buff_size]  = 1.0;

    //solve with 4th-order Runge Kutta
    for(int i=buff_size; i<lattice_size_buff-1; ++i){

        double r, A, rho;

        Bi_Linears bi_linears, bi_linears_midpoint, bi_linears_plus;


        double k1_A = 0.0, k2_A = 0.0, k3_A = 0.0, k4_A = 0.0;

        r = (i-buff_size)*dr;

        A = metric->A[i];

        set_bi_linears(i,           &bi_linears,            c_fields, q_fields, metric);
        set_bi_linears_midpoint(i,  &bi_linears_midpoint,   c_fields, q_fields, metric);
        set_bi_linears(i+1,         &bi_linears_plus,       c_fields, q_fields, metric);


        k1_A            = dr*set_dA_dr_initial(r,           A,  bi_linears);

        k2_A            = dr*set_dA_dr_initial(r+0.5*dr,    A+0.5*k1_A,  bi_linears);

        k3_A            = dr*set_dA_dr_initial(r+0.5*dr,    A+0.5*k2_A,  bi_linears);

        k4_A            = dr*set_dA_dr_initial(r+dr,        A+k3_A,  bi_linears);

        metric->A[i+1]  = metric->A[i]  + ( k1_A + 2.0*k2_A + 2.0*k3_A + k4_A ) / 6.0;


    }

    //Now we rescale pi and find the initial U_tilda and lambda
    for(int i=buff_size; i<lattice_size_buff; ++i){
        double r;

        r = (i-buff_size)*dr;
        metric->lambda[i] = (i!=buff_size ? 1.0/r*(1.0-metric->A[i]/metric->B[i]) : 0.0);

        //metric->U_tilda[i]=1.0/r*(1.0-metric->A[i])*(1.0-4.0/metric->A[i])+r/2.0*M_P*M_P*rho;
        metric->U_tilda[i] = (first_deriv(i, metric->A)-4.0*metric->lambda[i])/metric->A[i];

        c_fields->pi[i]    = sqrt(metric->A[i])*c_fields->pi[i];                  //here we rescale the placeholder pi

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
                    q_fields[which_q_field]->pi[k][l][i]         = sqrt(metric->A[i])*(q_fields[which_q_field]->pi[k][l][i]);         //here we rescale the placeholder pi_mode

                }
            }
        }

    }
    metric->lambda[buff_size]  = 0.0;
    metric->U_tilda[buff_size] = 0.0;

    set_buff_zone(c_fields, q_fields, metric);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Time iteration equations for all of the background variable fields */   
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void df_dt(double cos_const, __complex__ double ****alpha_AB_pi_mode, Classical_fields *c_fields, Classical_fields *c_fields_dot, Quantum_fields **q_fields, Quantum_fields **q_fields_dot, Metric_Fields *metric, Metric_Fields *metric_dot){

	

	double alpha_AB_pi[lattice_size_buff], alpha_BA_chi[lattice_size_buff], K_B_alpha[lattice_size_buff], alpha_K[lattice_size_buff];
    double alpha_B_over_A[lattice_size_buff];
    double alpha_AB_q_pi[lattice_size_buff], alpha_BA_q_chi[lattice_size_buff];

    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for (int m=0; m<lattice_size_buff; m++){
        alpha_AB_pi[m]    =   metric->alpha[m]/(metric->B[m]*sqrt(metric->A[m]))*c_fields->pi[m];
        alpha_BA_chi[m]   =   metric->alpha[m]*metric->B[m]/(sqrt(metric->A[m]))*c_fields->chi[m];
        alpha_AB_q_pi[m]    =   metric->alpha[m]/(metric->B[m]*sqrt(metric->A[m]))*c_fields->q_pi[m];
        alpha_BA_q_chi[m]   =   metric->alpha[m]*metric->B[m]/(sqrt(metric->A[m]))*c_fields->q_chi[m];
        K_B_alpha[m]      =   metric->alpha[m]*metric->K_B[m];
        alpha_K[m]        =   metric->alpha[m]*metric->K[m];
        alpha_B_over_A[m] =   metric->alpha[m]*metric->B[m]/(sqrt(metric->A[m]));

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
                    alpha_AB_pi_mode[which_q_field][k][l][m]=metric->alpha[m]/(metric->B[m]*sqrt(metric->A[m]))*q_fields[which_q_field]->pi[k][l][m];
                }
            }
        }
    }
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size)
    for (int i=buff_size; i<lattice_size_buff; i++){

        Stress_Tensor stress_tnsr;
        Bi_Linears    bi_linears;

        double rho = 0.0, j_A = 0.0, S_A = 0.0, S_B = 0.0;

        double A, B, D_B, U_tilda, K, K_B, lambda, alpha, D_alpha;

        double phi, chi, pi, q_phi, q_chi, q_pi;

        double r, rpoints[lattice_size_buff];
        make_points(rpoints);
        r=rpoints[i];

        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric);
        set_stress_tensor(i, &stress_tnsr, bi_linears, metric, c_fields, q_fields);

        rho = stress_tnsr.rho;
        j_A = stress_tnsr.j_A;
        S_A = stress_tnsr.S_A;
        S_B = stress_tnsr.S_B;

        A       = metric->A[i];
        B       = metric->B[i];
        D_B     = metric->D_B[i];
        U_tilda = metric->U_tilda[i];
        K       = metric->K[i];
        K_B     = metric->K_B[i];
        lambda  = metric->lambda[i];
        alpha   = metric->alpha[i];
        D_alpha = metric->D_alpha[i];


        phi = c_fields->phi[i];
        chi = c_fields->chi[i];
        pi  = c_fields->pi[i];

        q_phi = c_fields->q_phi[i];
        q_chi = c_fields->q_chi[i];
        q_pi  = c_fields->q_pi[i];


        /* Actual evolution equations*/
        
        metric_dot->A[i]       = (-2.0*alpha*A*(K-2.0*K_B));

        metric_dot->B[i]       =  (-2.0*alpha*B*K_B);

        metric_dot->D_B[i]     = -2.0*first_deriv(i, K_B_alpha);

        metric_dot->U_tilda[i] = -2.0*alpha*(first_deriv(i,metric->K)+D_alpha*(K-4.0*K_B)
                                                -2.0*(K-3.0*K_B)*(D_B-2.0*lambda*B/A))
                                    -4.0*alpha*j_A*(M_P*M_P);

        

        metric_dot->K_B[i]     =  (i!=buff_size ? alpha/(r*A)*(0.5*U_tilda + 2.0*lambda*B/A -D_B -lambda -D_alpha) : alpha/(A)*(0.5*first_deriv(i, metric->U_tilda)
                                                                                        +(2.0*B/A-1.0)*first_deriv(i, metric->lambda)
                                                                                        -first_deriv(i, metric->D_B)
                                                                                        -first_deriv(i, metric->D_alpha)))
                                    +alpha/A*(-0.5*D_alpha*D_B
                                            -0.5*first_deriv(i, metric->D_B)+0.25*D_B*(U_tilda+4.0*lambda*B/A)
                                            +A*K*K_B)
                                    +alpha/(2.0)*M_P*M_P*(S_A-rho+2.0*cos_const);   

       

        metric_dot->K[i] =  alpha*(K*K-4.0*K*K_B+6.0*K_B*K_B)
                                - (i != buff_size ? alpha/A*(first_deriv(i, metric->D_alpha) + D_alpha*D_alpha + 2.0*D_alpha/r -0.5*D_alpha*(U_tilda+4.0*lambda*B/A)) : alpha/A*(3.0*first_deriv(i, metric->D_alpha)))
                                    +alpha/2.0*M_P*M_P*(rho+S_A+2.0*S_B+2.0*cos_const);
        

        metric_dot->lambda[i]  = (i != buff_size ? 2.0*alpha*A/B*(first_deriv(i, metric->K_B)-0.5*D_B*(K-3.0*K_B)+1.0/2.0*M_P*M_P*j_A) : 0.0);
        

        
        metric_dot->alpha[i]   = -2.0*alpha*K;                                                      ///// 1+LOG GAUGE
        metric_dot->D_alpha[i] = (i != buff_size ? -2.0*first_deriv(i, metric->K) : 0.0);           ///// 1+LOG GAUGE

        //metric_dot->alpha[i]   = -alpha*alpha*K;                             ///// HARMONIC GAUGE
        //metric_dot->D_alpha[i] = -first_deriv(i, alpha_K);                   ///// HARMONIC GAUGE

        c_fields_dot->phi[i] = alpha/(sqrt(A)*B)*pi;
        c_fields_dot->chi[i] = first_deriv(i, alpha_AB_pi);   ////
        c_fields_dot->pi[i]  = (i!=buff_size ? first_deriv(i, alpha_BA_chi) +2.0/r*alpha*B/sqrt(A)*chi : 3.0*alpha*B/sqrt(A)*first_deriv(i, c_fields->chi));
        
        c_fields_dot->q_phi[i] = alpha/(sqrt(A)*B)*q_pi;
        c_fields_dot->q_chi[i] = first_deriv(i, alpha_AB_q_pi);   ////
        c_fields_dot->q_pi[i]  = (i!=buff_size ? first_deriv(i, alpha_BA_q_chi) +2.0/r*alpha*B/sqrt(A)*q_chi : 3.0*alpha*B/sqrt(A)*first_deriv(i, c_fields->q_chi));
        
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                int l_value;
                l_value = l_start + l*l_step;
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
                    __complex__ double phi_mode=0.0, chi_mode=0.0, pi_mode=0.0;

                    phi_mode=q_fields[which_q_field]->phi[k][l][i];
                    chi_mode=q_fields[which_q_field]->chi[k][l][i];
                    pi_mode=q_fields[which_q_field]->pi[k][l][i];
                    
                    q_fields_dot[which_q_field]->phi[k][l][i] =  alpha / (sqrt(A) * B) * pi_mode;

                    q_fields_dot[which_q_field]->chi[k][l][i] = first_deriv_comp(i, alpha_AB_pi_mode[which_q_field][k][l]);

                    q_fields_dot[which_q_field]->pi[k][l][i] = (first_deriv(i, alpha_B_over_A) * (i != buff_size ? (l_value / r * phi_mode + chi_mode) : 0.0)

                        + alpha * B / sqrt(A) * (i != buff_size ? (first_deriv_comp(i, q_fields[which_q_field]->chi[k][l]) + (2.0 * l_value + 2.0) / r * chi_mode) : ((2.0 * l_value + 3.0) * first_deriv_comp(i, q_fields[which_q_field]->chi[k][l])))
               
                                              + (i != buff_size ? l_value*(l_value+1)/r*alpha*B/sqrt(A)*lambda*phi_mode : l_value*(l_value+1)*alpha*B/sqrt(A)*phi_mode*first_deriv(i, metric->lambda))
                        - alpha*B*sqrt(A)*massSq[which_q_field] * phi_mode);

                    // MINKOWSKI VERSION
                    //q_fields_dot[which_q_field]->phi[k][l][i] = pi_mode;
                    //q_fields_dot[which_q_field]->chi[k][l][i] = first_deriv_comp(i, q_fields[which_q_field]->pi[k][l]);
                    //q_fields_dot[which_q_field]->pi[k][l][i] = (i != buff_size ? (first_deriv_comp(i, q_fields[which_q_field]->chi[k][l]) + (2.0*l_value+2.0)/r* chi_mode - massSq[which_q_field]*phi_mode) : ((2.0*l_value+3.0) * first_deriv_comp(i, q_fields[which_q_field]->chi[k][l])- massSq[which_q_field] * phi_mode));
                }
            }
        }


    }


     /* Now add the damping term*/
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size)
    for (int i = buff_size; i < lattice_size_buff; i++) {
       
        double r;
        r = (i - buff_size) * dr;

        if (damping_order == 12) {
            
            metric_dot->A[i]        = (metric->alpha[i] != 0.0 ? metric_dot->A[i]       - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->A)        : metric_dot->A[i]);
            metric_dot->B[i]        = (metric->alpha[i] != 0.0 ? metric_dot->B[i]       - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->B)        : metric_dot->B[i]);
            metric_dot->D_B[i]      = (metric->alpha[i] != 0.0 ? metric_dot->D_B[i]     - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->D_B)      : metric_dot->D_B[i]);
            metric_dot->U_tilda[i]  = (metric->alpha[i] != 0.0 ? metric_dot->U_tilda[i] - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->U_tilda)  : metric_dot->U_tilda[i]);
            metric_dot->K[i]        = (metric->alpha[i] != 0.0 ? metric_dot->K[i]       - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->K)        : metric_dot->K[i]);
            metric_dot->K_B[i]      = (metric->alpha[i] != 0.0 ? metric_dot->K_B[i]     - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->K_B)      : metric_dot->K_B[i]);
            metric_dot->lambda[i]   = (metric->alpha[i] != 0.0 ? metric_dot->lambda[i]  - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->lambda)   : metric_dot->lambda[i]);
            metric_dot->alpha[i]    = (metric->alpha[i] != 0.0 ? metric_dot->alpha[i]   - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->alpha)    : metric_dot->alpha[i]);
            metric_dot->D_alpha[i]  = (metric->alpha[i] != 0.0 ? metric_dot->D_alpha[i] - epsilonc * pow(dr, 11) * twelfth_deriv(i, metric->D_alpha)  : metric_dot->D_alpha[i]);

            c_fields_dot->phi[i] = (metric->alpha[i] != 0.0 ? c_fields_dot->phi[i]      - epsilonc * pow(dr, 11) * twelfth_deriv(i, c_fields->phi)    : c_fields_dot->phi[i]);
            c_fields_dot->chi[i] = (metric->alpha[i] != 0.0 ? c_fields_dot->chi[i]      - epsilonc * pow(dr, 11) * twelfth_deriv(i, c_fields->chi)    : c_fields_dot->chi[i]);
            c_fields_dot->pi[i]  = (metric->alpha[i] != 0.0 ? c_fields_dot->pi[i]       - epsilonc * pow(dr, 11) * twelfth_deriv(i, c_fields->pi)     : c_fields_dot->pi[i]);

            c_fields_dot->q_phi[i] = (metric->alpha[i] != 0.0 ? c_fields_dot->q_phi[i]      - epsilonc * pow(dr, 11) * twelfth_deriv(i, c_fields->q_phi)    : c_fields_dot->q_phi[i]);
            c_fields_dot->q_chi[i] = (metric->alpha[i] != 0.0 ? c_fields_dot->q_chi[i]      - epsilonc * pow(dr, 11) * twelfth_deriv(i, c_fields->q_chi)    : c_fields_dot->q_chi[i]);
            c_fields_dot->q_pi[i]  = (metric->alpha[i] != 0.0 ? c_fields_dot->q_pi[i]       - epsilonc * pow(dr, 11) * twelfth_deriv(i, c_fields->q_pi)     : c_fields_dot->q_pi[i]);

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                        q_fields_dot[which_q_field]->phi[k][l][i] = (metric->alpha[i] != 0.0 ? q_fields_dot[which_q_field]->phi[k][l][i] - epsilonq * pow(dr, 11) * twelfth_deriv_comp(i, q_fields[which_q_field]->phi[k][l]) : q_fields_dot[which_q_field]->phi[k][l][i]);
                        q_fields_dot[which_q_field]->chi[k][l][i] = (metric->alpha[i] != 0.0 ? q_fields_dot[which_q_field]->chi[k][l][i] - epsilonq * pow(dr, 11) * twelfth_deriv_comp(i, q_fields[which_q_field]->chi[k][l]) : q_fields_dot[which_q_field]->chi[k][l][i]);
                        q_fields_dot[which_q_field]->pi[k][l][i]  = (metric->alpha[i] != 0.0 ? q_fields_dot[which_q_field]->pi[k][l][i]  - epsilonq * pow(dr, 11) * twelfth_deriv_comp(i, q_fields[which_q_field]->pi[k][l])  : q_fields_dot[which_q_field]->pi[k][l][i]);
                    }
                }
            }
        }
        else if (damping_order == 4) {
            
            metric_dot->A[i]        = (metric->alpha[i] != 0.0 ? metric_dot->A[i]       - exp(-r * r * 0.0 / 5.0)*epsilonc * pow(dr, 3) * fourth_deriv(i, metric->A)        : metric_dot->A[i]);
            metric_dot->B[i]        = (metric->alpha[i] != 0.0 ? metric_dot->B[i]       - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->B)        : metric_dot->B[i]);
            metric_dot->D_B[i]      = (metric->alpha[i] != 0.0 ? metric_dot->D_B[i]     - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->D_B)      : metric_dot->D_B[i]);
            metric_dot->U_tilda[i]  = (metric->alpha[i] != 0.0 ? metric_dot->U_tilda[i] - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->U_tilda)  : metric_dot->U_tilda[i]);
            metric_dot->K[i]        = (metric->alpha[i] != 0.0 ? metric_dot->K[i]       - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->K)        : metric_dot->K[i]);
            metric_dot->K_B[i]      = (metric->alpha[i] != 0.0 ? metric_dot->K_B[i]     - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->K_B)      : metric_dot->K_B[i]);
            metric_dot->lambda[i]   = (metric->alpha[i] != 0.0 ? metric_dot->lambda[i]  - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->lambda)   : metric_dot->lambda[i]);
            metric_dot->alpha[i]    = (metric->alpha[i] != 0.0 ? metric_dot->alpha[i]   - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->alpha)    : metric_dot->alpha[i]);
            metric_dot->D_alpha[i]  = (metric->alpha[i] != 0.0 ? metric_dot->D_alpha[i] - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, metric->D_alpha)  : metric_dot->D_alpha[i]);

            c_fields_dot->phi[i]    = (metric->alpha[i] != 0.0 ? c_fields_dot->phi[i]   - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, c_fields->phi)    : c_fields_dot->phi[i]);
            c_fields_dot->chi[i]    = (metric->alpha[i] != 0.0 ? c_fields_dot->chi[i]   - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, c_fields->chi)    : c_fields_dot->chi[i]);
            c_fields_dot->pi[i]     = (metric->alpha[i] != 0.0 ? c_fields_dot->pi[i]    - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, c_fields->pi)     : c_fields_dot->pi[i]);
            
            c_fields_dot->q_phi[i]    = (metric->alpha[i] != 0.0 ? c_fields_dot->q_phi[i]   - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, c_fields->q_phi)    : c_fields_dot->q_phi[i]);
            c_fields_dot->q_chi[i]    = (metric->alpha[i] != 0.0 ? c_fields_dot->q_chi[i]   - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, c_fields->q_chi)    : c_fields_dot->q_chi[i]);
            c_fields_dot->q_pi[i]     = (metric->alpha[i] != 0.0 ? c_fields_dot->q_pi[i]    - exp(-r * r * 0.0 / 5.0) * epsilonc * pow(dr, 3) * fourth_deriv(i, c_fields->q_pi)     : c_fields_dot->q_pi[i]);
            
            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {

                        q_fields_dot[which_q_field]->phi[k][l][i] = (metric->alpha[i] != -1.0 ? q_fields_dot[which_q_field]->phi[k][l][i] - exp(-r * r * 0.0 / 5.0) * epsilonq * pow(dr, 3) * fourth_deriv_comp(i, q_fields[which_q_field]->phi[k][l]) : q_fields_dot[which_q_field]->phi[k][l][i]);
                        q_fields_dot[which_q_field]->chi[k][l][i] = (metric->alpha[i] != -1.0 ? q_fields_dot[which_q_field]->chi[k][l][i] - exp(-r * r * 0.0 / 5.0) * epsilonq * pow(dr, 3) * fourth_deriv_comp(i, q_fields[which_q_field]->chi[k][l]) : q_fields_dot[which_q_field]->chi[k][l][i]);
                        q_fields_dot[which_q_field]->pi[k][l][i]  = (metric->alpha[i] != -1.0 ? q_fields_dot[which_q_field]->pi[k][l][i]  - exp(-r * r * 0.0 / 5.0) * epsilonq * pow(dr, 3) * fourth_deriv_comp(i, q_fields[which_q_field]->pi[k][l])  : q_fields_dot[which_q_field]->pi[k][l][i]);
                    }
                }
            }
        }

    }
    /* SET BUFF ZONE */
    set_buff_zone(c_fields_dot, q_fields_dot, metric_dot);
    

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* One step of the very accurate time iteration scheme */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void single_RK_convergence_step_RK5(double a_ij[nu_legendre][nu_legendre], __complex__ double**** alpha_AB_pi_mode, Classical_fields *c_fields_RK1, Classical_fields *c_fields_RK2, Classical_fields *c_fields_RK3,
                                    Classical_fields *c_fields_RK4, Classical_fields *c_fields_RK5, Classical_fields *c_fields_RK_sum, Classical_fields *c_fields,
                                    Quantum_fields **q_fields_RK1, Quantum_fields **q_fields_RK2, Quantum_fields **q_fields_RK3, Quantum_fields **q_fields_RK4,
                                    Quantum_fields **q_fields_RK5, Quantum_fields **q_fields_RK_sum, Quantum_fields **q_fields,
                                    Metric_Fields *metric_RK1, Metric_Fields *metric_RK2,    Metric_Fields *metric_RK3, Metric_Fields *metric_RK4,
                                    Metric_Fields *metric_RK5, Metric_Fields *metric_RK_sum, Metric_Fields *metric, double cos_const){

    //int l_value;

    //first iterate the RK1 term
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for(int i=0; i<lattice_size_buff; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[0][0]*c_fields_RK1->phi[i] + a_ij[0][1]*c_fields_RK2->phi[i]+a_ij[0][2]*c_fields_RK3->phi[i]+a_ij[0][3]*c_fields_RK4->phi[i]+a_ij[0][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[0][0]*c_fields_RK1->pi[i]  + a_ij[0][1]*c_fields_RK2->pi[i] +a_ij[0][2]*c_fields_RK3->pi[i] +a_ij[0][3]*c_fields_RK4->pi[i] +a_ij[0][4]*c_fields_RK5->pi[i]);
        c_fields_RK_sum->chi[i]= c_fields->chi[i]+ dt*(a_ij[0][0]*c_fields_RK1->chi[i] + a_ij[0][1]*c_fields_RK2->chi[i]+a_ij[0][2]*c_fields_RK3->chi[i]+a_ij[0][3]*c_fields_RK4->chi[i]+a_ij[0][4]*c_fields_RK5->chi[i]);

        c_fields_RK_sum->q_phi[i]= c_fields->q_phi[i]+ dt*(a_ij[0][0]*c_fields_RK1->q_phi[i] + a_ij[0][1]*c_fields_RK2->q_phi[i]+a_ij[0][2]*c_fields_RK3->q_phi[i]+a_ij[0][3]*c_fields_RK4->q_phi[i]+a_ij[0][4]*c_fields_RK5->q_phi[i]);
        c_fields_RK_sum->q_pi[i] = c_fields->q_pi[i] + dt*(a_ij[0][0]*c_fields_RK1->q_pi[i]  + a_ij[0][1]*c_fields_RK2->q_pi[i] +a_ij[0][2]*c_fields_RK3->q_pi[i] +a_ij[0][3]*c_fields_RK4->q_pi[i] +a_ij[0][4]*c_fields_RK5->q_pi[i]);
        c_fields_RK_sum->q_chi[i]= c_fields->q_chi[i]+ dt*(a_ij[0][0]*c_fields_RK1->q_chi[i] + a_ij[0][1]*c_fields_RK2->q_chi[i]+a_ij[0][2]*c_fields_RK3->q_chi[i]+a_ij[0][3]*c_fields_RK4->q_chi[i]+a_ij[0][4]*c_fields_RK5->q_chi[i]);

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[0][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[0][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[0][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[0][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[0][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->chi[k][l][i]   = q_fields[which_q_field]->chi[k][l][i] + dt*(a_ij[0][0]*q_fields_RK1[which_q_field]->chi[k][l][i] + a_ij[0][1]*q_fields_RK2[which_q_field]->chi[k][l][i]
                                                                                                                +a_ij[0][2]*q_fields_RK3[which_q_field]->chi[k][l][i] + a_ij[0][3]*q_fields_RK4[which_q_field]->chi[k][l][i]+a_ij[0][4]*q_fields_RK5[which_q_field]->chi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[0][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[0][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[0][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[0][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[0][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[0][0]*metric_RK1->A[i]      + a_ij[0][1]*metric_RK2->A[i]        + a_ij[0][2]*metric_RK3->A[i]       + a_ij[0][3]*metric_RK4->A[i]       + a_ij[0][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[0][0]*metric_RK1->B[i]      + a_ij[0][1]*metric_RK2->B[i]        + a_ij[0][2]*metric_RK3->B[i]       + a_ij[0][3]*metric_RK4->B[i]       + a_ij[0][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[0][0]*metric_RK1->D_B[i]    + a_ij[0][1]*metric_RK2->D_B[i]      + a_ij[0][2]*metric_RK3->D_B[i]     + a_ij[0][3]*metric_RK4->D_B[i]     + a_ij[0][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[0][0]*metric_RK1->U_tilda[i]+ a_ij[0][1]*metric_RK2->U_tilda[i]  + a_ij[0][2]*metric_RK3->U_tilda[i] + a_ij[0][3]*metric_RK4->U_tilda[i] + a_ij[0][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[0][0]*metric_RK1->K[i]      + a_ij[0][1]*metric_RK2->K[i]        + a_ij[0][2]*metric_RK3->K[i]       + a_ij[0][3]*metric_RK4->K[i]       + a_ij[0][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[0][0]*metric_RK1->K_B[i]    + a_ij[0][1]*metric_RK2->K_B[i]      + a_ij[0][2]*metric_RK3->K_B[i]     + a_ij[0][3]*metric_RK4->K_B[i]     + a_ij[0][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[0][0]*metric_RK1->lambda[i] + a_ij[0][1]*metric_RK2->lambda[i]   + a_ij[0][2]*metric_RK3->lambda[i]  + a_ij[0][3]*metric_RK4->lambda[i]  + a_ij[0][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[0][0]*metric_RK1->alpha[i]  + a_ij[0][1]*metric_RK2->alpha[i]    + a_ij[0][2]*metric_RK3->alpha[i]   + a_ij[0][3]*metric_RK4->alpha[i]   + a_ij[0][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[0][0]*metric_RK1->D_alpha[i]+ a_ij[0][1]*metric_RK2->D_alpha[i]  + a_ij[0][2]*metric_RK3->D_alpha[i] + a_ij[0][3]*metric_RK4->D_alpha[i] + a_ij[0][4]*metric_RK5->D_alpha[i]);


    }
    
    df_dt(cos_const, alpha_AB_pi_mode, c_fields_RK_sum, c_fields_RK1, q_fields_RK_sum, q_fields_RK1, metric_RK_sum, metric_RK1);
    
    //then iterate the RK2 term
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for(int i=0; i<lattice_size_buff; ++i){

        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[1][0]*c_fields_RK1->phi[i] + a_ij[1][1]*c_fields_RK2->phi[i]+a_ij[1][2]*c_fields_RK3->phi[i]+a_ij[1][3]*c_fields_RK4->phi[i]+a_ij[1][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[1][0]*c_fields_RK1->pi[i]  + a_ij[1][1]*c_fields_RK2->pi[i] +a_ij[1][2]*c_fields_RK3->pi[i] +a_ij[1][3]*c_fields_RK4->pi[i] +a_ij[1][4]*c_fields_RK5->pi[i]);
        c_fields_RK_sum->chi[i]= c_fields->chi[i]+ dt*(a_ij[1][0]*c_fields_RK1->chi[i] + a_ij[1][1]*c_fields_RK2->chi[i]+a_ij[1][2]*c_fields_RK3->chi[i]+a_ij[1][3]*c_fields_RK4->chi[i]+a_ij[1][4]*c_fields_RK5->chi[i]);

        c_fields_RK_sum->q_phi[i]= c_fields->q_phi[i]+ dt*(a_ij[1][0]*c_fields_RK1->q_phi[i] + a_ij[1][1]*c_fields_RK2->q_phi[i]+a_ij[1][2]*c_fields_RK3->q_phi[i]+a_ij[1][3]*c_fields_RK4->q_phi[i]+a_ij[1][4]*c_fields_RK5->q_phi[i]);
        c_fields_RK_sum->q_pi[i] = c_fields->q_pi[i] + dt*(a_ij[1][0]*c_fields_RK1->q_pi[i]  + a_ij[1][1]*c_fields_RK2->q_pi[i] +a_ij[1][2]*c_fields_RK3->q_pi[i] +a_ij[1][3]*c_fields_RK4->q_pi[i] +a_ij[1][4]*c_fields_RK5->q_pi[i]);
        c_fields_RK_sum->q_chi[i]= c_fields->q_chi[i]+ dt*(a_ij[1][0]*c_fields_RK1->q_chi[i] + a_ij[1][1]*c_fields_RK2->q_chi[i]+a_ij[1][2]*c_fields_RK3->q_chi[i]+a_ij[1][3]*c_fields_RK4->q_chi[i]+a_ij[1][4]*c_fields_RK5->q_chi[i]);

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[1][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[1][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[1][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[1][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[1][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->chi[k][l][i]   = q_fields[which_q_field]->chi[k][l][i] + dt*(a_ij[1][0]*q_fields_RK1[which_q_field]->chi[k][l][i] + a_ij[1][1]*q_fields_RK2[which_q_field]->chi[k][l][i]
                                                                                                                +a_ij[1][2]*q_fields_RK3[which_q_field]->chi[k][l][i] + a_ij[1][3]*q_fields_RK4[which_q_field]->chi[k][l][i]+a_ij[1][4]*q_fields_RK5[which_q_field]->chi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[1][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[1][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[1][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[1][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[1][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[1][0]*metric_RK1->A[i]      + a_ij[1][1]*metric_RK2->A[i]        + a_ij[1][2]*metric_RK3->A[i]       + a_ij[1][3]*metric_RK4->A[i]       + a_ij[1][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[1][0]*metric_RK1->B[i]      + a_ij[1][1]*metric_RK2->B[i]        + a_ij[1][2]*metric_RK3->B[i]       + a_ij[1][3]*metric_RK4->B[i]       + a_ij[1][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[1][0]*metric_RK1->D_B[i]    + a_ij[1][1]*metric_RK2->D_B[i]      + a_ij[1][2]*metric_RK3->D_B[i]     + a_ij[1][3]*metric_RK4->D_B[i]     + a_ij[1][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[1][0]*metric_RK1->U_tilda[i]+ a_ij[1][1]*metric_RK2->U_tilda[i]  + a_ij[1][2]*metric_RK3->U_tilda[i] + a_ij[1][3]*metric_RK4->U_tilda[i] + a_ij[1][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[1][0]*metric_RK1->K[i]      + a_ij[1][1]*metric_RK2->K[i]        + a_ij[1][2]*metric_RK3->K[i]       + a_ij[1][3]*metric_RK4->K[i]       + a_ij[1][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[1][0]*metric_RK1->K_B[i]    + a_ij[1][1]*metric_RK2->K_B[i]      + a_ij[1][2]*metric_RK3->K_B[i]     + a_ij[1][3]*metric_RK4->K_B[i]     + a_ij[1][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[1][0]*metric_RK1->lambda[i] + a_ij[1][1]*metric_RK2->lambda[i]   + a_ij[1][2]*metric_RK3->lambda[i]  + a_ij[1][3]*metric_RK4->lambda[i]  + a_ij[1][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[1][0]*metric_RK1->alpha[i]  + a_ij[1][1]*metric_RK2->alpha[i]    + a_ij[1][2]*metric_RK3->alpha[i]   + a_ij[1][3]*metric_RK4->alpha[i]   + a_ij[1][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[1][0]*metric_RK1->D_alpha[i]+ a_ij[1][1]*metric_RK2->D_alpha[i]  + a_ij[1][2]*metric_RK3->D_alpha[i] + a_ij[1][3]*metric_RK4->D_alpha[i] + a_ij[1][4]*metric_RK5->D_alpha[i]);

    }

    df_dt(cos_const, alpha_AB_pi_mode, c_fields_RK_sum, c_fields_RK2, q_fields_RK_sum, q_fields_RK2, metric_RK_sum, metric_RK2);

    //then iterate the RK3 term
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for(int i=0; i<lattice_size_buff; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[2][0]*c_fields_RK1->phi[i] + a_ij[2][1]*c_fields_RK2->phi[i]+a_ij[2][2]*c_fields_RK3->phi[i]+a_ij[2][3]*c_fields_RK4->phi[i]+a_ij[2][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[2][0]*c_fields_RK1->pi[i]  + a_ij[2][1]*c_fields_RK2->pi[i] +a_ij[2][2]*c_fields_RK3->pi[i] +a_ij[2][3]*c_fields_RK4->pi[i] +a_ij[2][4]*c_fields_RK5->pi[i]);
        c_fields_RK_sum->chi[i]= c_fields->chi[i]+ dt*(a_ij[2][0]*c_fields_RK1->chi[i] + a_ij[2][1]*c_fields_RK2->chi[i]+a_ij[2][2]*c_fields_RK3->chi[i]+a_ij[2][3]*c_fields_RK4->chi[i]+a_ij[2][4]*c_fields_RK5->chi[i]);

        c_fields_RK_sum->q_phi[i]= c_fields->q_phi[i]+ dt*(a_ij[2][0]*c_fields_RK1->q_phi[i] + a_ij[2][1]*c_fields_RK2->q_phi[i]+a_ij[2][2]*c_fields_RK3->q_phi[i]+a_ij[2][3]*c_fields_RK4->q_phi[i]+a_ij[2][4]*c_fields_RK5->q_phi[i]);
        c_fields_RK_sum->q_pi[i] = c_fields->q_pi[i] + dt*(a_ij[2][0]*c_fields_RK1->q_pi[i]  + a_ij[2][1]*c_fields_RK2->q_pi[i] +a_ij[2][2]*c_fields_RK3->q_pi[i] +a_ij[2][3]*c_fields_RK4->q_pi[i] +a_ij[2][4]*c_fields_RK5->q_pi[i]);
        c_fields_RK_sum->q_chi[i]= c_fields->q_chi[i]+ dt*(a_ij[2][0]*c_fields_RK1->q_chi[i] + a_ij[2][1]*c_fields_RK2->q_chi[i]+a_ij[2][2]*c_fields_RK3->q_chi[i]+a_ij[2][3]*c_fields_RK4->q_chi[i]+a_ij[2][4]*c_fields_RK5->q_chi[i]);

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[2][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[2][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[2][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[2][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[2][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->chi[k][l][i]   = q_fields[which_q_field]->chi[k][l][i] + dt*(a_ij[2][0]*q_fields_RK1[which_q_field]->chi[k][l][i] + a_ij[2][1]*q_fields_RK2[which_q_field]->chi[k][l][i]
                                                                                                                +a_ij[2][2]*q_fields_RK3[which_q_field]->chi[k][l][i] + a_ij[2][3]*q_fields_RK4[which_q_field]->chi[k][l][i]+a_ij[2][4]*q_fields_RK5[which_q_field]->chi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[2][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[2][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[2][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[2][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[2][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[2][0]*metric_RK1->A[i]      + a_ij[2][1]*metric_RK2->A[i]        + a_ij[2][2]*metric_RK3->A[i]       + a_ij[2][3]*metric_RK4->A[i]       + a_ij[2][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[2][0]*metric_RK1->B[i]      + a_ij[2][1]*metric_RK2->B[i]        + a_ij[2][2]*metric_RK3->B[i]       + a_ij[2][3]*metric_RK4->B[i]       + a_ij[2][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[2][0]*metric_RK1->D_B[i]    + a_ij[2][1]*metric_RK2->D_B[i]      + a_ij[2][2]*metric_RK3->D_B[i]     + a_ij[2][3]*metric_RK4->D_B[i]     + a_ij[2][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[2][0]*metric_RK1->U_tilda[i]+ a_ij[2][1]*metric_RK2->U_tilda[i]  + a_ij[2][2]*metric_RK3->U_tilda[i] + a_ij[2][3]*metric_RK4->U_tilda[i] + a_ij[2][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[2][0]*metric_RK1->K[i]      + a_ij[2][1]*metric_RK2->K[i]        + a_ij[2][2]*metric_RK3->K[i]       + a_ij[2][3]*metric_RK4->K[i]       + a_ij[2][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[2][0]*metric_RK1->K_B[i]    + a_ij[2][1]*metric_RK2->K_B[i]      + a_ij[2][2]*metric_RK3->K_B[i]     + a_ij[2][3]*metric_RK4->K_B[i]     + a_ij[2][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[2][0]*metric_RK1->lambda[i] + a_ij[2][1]*metric_RK2->lambda[i]   + a_ij[2][2]*metric_RK3->lambda[i]  + a_ij[2][3]*metric_RK4->lambda[i]  + a_ij[2][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[2][0]*metric_RK1->alpha[i]  + a_ij[2][1]*metric_RK2->alpha[i]    + a_ij[2][2]*metric_RK3->alpha[i]   + a_ij[2][3]*metric_RK4->alpha[i]   + a_ij[2][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[2][0]*metric_RK1->D_alpha[i]+ a_ij[2][1]*metric_RK2->D_alpha[i]  + a_ij[2][2]*metric_RK3->D_alpha[i] + a_ij[2][3]*metric_RK4->D_alpha[i] + a_ij[2][4]*metric_RK5->D_alpha[i]);

    }
    df_dt(cos_const, alpha_AB_pi_mode, c_fields_RK_sum, c_fields_RK3, q_fields_RK_sum, q_fields_RK3, metric_RK_sum, metric_RK3);
    //then iterate the RK4 term
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for(int i=0; i<lattice_size_buff; ++i){

        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[3][0]*c_fields_RK1->phi[i] + a_ij[3][1]*c_fields_RK2->phi[i]+a_ij[3][2]*c_fields_RK3->phi[i]+a_ij[3][3]*c_fields_RK4->phi[i]+a_ij[3][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[3][0]*c_fields_RK1->pi[i]  + a_ij[3][1]*c_fields_RK2->pi[i] +a_ij[3][2]*c_fields_RK3->pi[i] +a_ij[3][3]*c_fields_RK4->pi[i] +a_ij[3][4]*c_fields_RK5->pi[i]);
        c_fields_RK_sum->chi[i]= c_fields->chi[i]+ dt*(a_ij[3][0]*c_fields_RK1->chi[i] + a_ij[3][1]*c_fields_RK2->chi[i]+a_ij[3][2]*c_fields_RK3->chi[i]+a_ij[3][3]*c_fields_RK4->chi[i]+a_ij[3][4]*c_fields_RK5->chi[i]);

        c_fields_RK_sum->q_phi[i]= c_fields->q_phi[i]+ dt*(a_ij[3][0]*c_fields_RK1->q_phi[i] + a_ij[3][1]*c_fields_RK2->q_phi[i]+a_ij[3][2]*c_fields_RK3->q_phi[i]+a_ij[3][3]*c_fields_RK4->q_phi[i]+a_ij[3][4]*c_fields_RK5->q_phi[i]);
        c_fields_RK_sum->q_pi[i] = c_fields->q_pi[i] + dt*(a_ij[3][0]*c_fields_RK1->q_pi[i]  + a_ij[3][1]*c_fields_RK2->q_pi[i] +a_ij[3][2]*c_fields_RK3->q_pi[i] +a_ij[3][3]*c_fields_RK4->q_pi[i] +a_ij[3][4]*c_fields_RK5->q_pi[i]);
        c_fields_RK_sum->q_chi[i]= c_fields->q_chi[i]+ dt*(a_ij[3][0]*c_fields_RK1->q_chi[i] + a_ij[3][1]*c_fields_RK2->q_chi[i]+a_ij[3][2]*c_fields_RK3->q_chi[i]+a_ij[3][3]*c_fields_RK4->q_chi[i]+a_ij[3][4]*c_fields_RK5->q_chi[i]);

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[3][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[3][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[3][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[3][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[3][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->chi[k][l][i]   = q_fields[which_q_field]->chi[k][l][i] + dt*(a_ij[3][0]*q_fields_RK1[which_q_field]->chi[k][l][i] + a_ij[3][1]*q_fields_RK2[which_q_field]->chi[k][l][i]
                                                                                                                +a_ij[3][2]*q_fields_RK3[which_q_field]->chi[k][l][i] + a_ij[3][3]*q_fields_RK4[which_q_field]->chi[k][l][i]+a_ij[3][4]*q_fields_RK5[which_q_field]->chi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[3][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[3][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[3][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[3][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[3][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[3][0]*metric_RK1->A[i]      + a_ij[3][1]*metric_RK2->A[i]        + a_ij[3][2]*metric_RK3->A[i]       + a_ij[3][3]*metric_RK4->A[i]       + a_ij[3][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[3][0]*metric_RK1->B[i]      + a_ij[3][1]*metric_RK2->B[i]        + a_ij[3][2]*metric_RK3->B[i]       + a_ij[3][3]*metric_RK4->B[i]       + a_ij[3][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[3][0]*metric_RK1->D_B[i]    + a_ij[3][1]*metric_RK2->D_B[i]      + a_ij[3][2]*metric_RK3->D_B[i]     + a_ij[3][3]*metric_RK4->D_B[i]     + a_ij[3][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[3][0]*metric_RK1->U_tilda[i]+ a_ij[3][1]*metric_RK2->U_tilda[i]  + a_ij[3][2]*metric_RK3->U_tilda[i] + a_ij[3][3]*metric_RK4->U_tilda[i] + a_ij[3][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[3][0]*metric_RK1->K[i]      + a_ij[3][1]*metric_RK2->K[i]        + a_ij[3][2]*metric_RK3->K[i]       + a_ij[3][3]*metric_RK4->K[i]       + a_ij[3][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[3][0]*metric_RK1->K_B[i]    + a_ij[3][1]*metric_RK2->K_B[i]      + a_ij[3][2]*metric_RK3->K_B[i]     + a_ij[3][3]*metric_RK4->K_B[i]     + a_ij[3][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[3][0]*metric_RK1->lambda[i] + a_ij[3][1]*metric_RK2->lambda[i]   + a_ij[3][2]*metric_RK3->lambda[i]  + a_ij[3][3]*metric_RK4->lambda[i]  + a_ij[3][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[3][0]*metric_RK1->alpha[i]  + a_ij[3][1]*metric_RK2->alpha[i]    + a_ij[3][2]*metric_RK3->alpha[i]   + a_ij[3][3]*metric_RK4->alpha[i]   + a_ij[3][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[3][0]*metric_RK1->D_alpha[i]+ a_ij[3][1]*metric_RK2->D_alpha[i]  + a_ij[3][2]*metric_RK3->D_alpha[i] + a_ij[3][3]*metric_RK4->D_alpha[i] + a_ij[3][4]*metric_RK5->D_alpha[i]);

    }
    df_dt(cos_const, alpha_AB_pi_mode, c_fields_RK_sum, c_fields_RK4, q_fields_RK_sum, q_fields_RK4, metric_RK_sum, metric_RK4);
    //then iterate the RK5 term
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for(int i=0; i<lattice_size_buff; ++i){
        c_fields_RK_sum->phi[i]= c_fields->phi[i]+ dt*(a_ij[4][0]*c_fields_RK1->phi[i] + a_ij[4][1]*c_fields_RK2->phi[i]+a_ij[4][2]*c_fields_RK3->phi[i]+a_ij[4][3]*c_fields_RK4->phi[i]+a_ij[4][4]*c_fields_RK5->phi[i]);
        c_fields_RK_sum->pi[i] = c_fields->pi[i] + dt*(a_ij[4][0]*c_fields_RK1->pi[i]  + a_ij[4][1]*c_fields_RK2->pi[i] +a_ij[4][2]*c_fields_RK3->pi[i] +a_ij[4][3]*c_fields_RK4->pi[i] +a_ij[4][4]*c_fields_RK5->pi[i]);
        c_fields_RK_sum->chi[i]= c_fields->chi[i]+ dt*(a_ij[4][0]*c_fields_RK1->chi[i] + a_ij[4][1]*c_fields_RK2->chi[i]+a_ij[4][2]*c_fields_RK3->chi[i]+a_ij[4][3]*c_fields_RK4->chi[i]+a_ij[4][4]*c_fields_RK5->chi[i]);

        c_fields_RK_sum->q_phi[i]= c_fields->q_phi[i]+ dt*(a_ij[4][0]*c_fields_RK1->q_phi[i] + a_ij[4][1]*c_fields_RK2->q_phi[i]+a_ij[4][2]*c_fields_RK3->q_phi[i]+a_ij[4][3]*c_fields_RK4->q_phi[i]+a_ij[4][4]*c_fields_RK5->q_phi[i]);
        c_fields_RK_sum->q_pi[i] = c_fields->q_pi[i] + dt*(a_ij[4][0]*c_fields_RK1->q_pi[i]  + a_ij[4][1]*c_fields_RK2->q_pi[i] +a_ij[4][2]*c_fields_RK3->q_pi[i] +a_ij[4][3]*c_fields_RK4->q_pi[i] +a_ij[4][4]*c_fields_RK5->q_pi[i]);
        c_fields_RK_sum->q_chi[i]= c_fields->q_chi[i]+ dt*(a_ij[4][0]*c_fields_RK1->q_chi[i] + a_ij[4][1]*c_fields_RK2->q_chi[i]+a_ij[4][2]*c_fields_RK3->q_chi[i]+a_ij[4][3]*c_fields_RK4->q_chi[i]+a_ij[4][4]*c_fields_RK5->q_chi[i]);

        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                    q_fields_RK_sum[which_q_field]->phi[k][l][i]   = q_fields[which_q_field]->phi[k][l][i] + dt*(a_ij[4][0]*q_fields_RK1[which_q_field]->phi[k][l][i] + a_ij[4][1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                +a_ij[4][2]*q_fields_RK3[which_q_field]->phi[k][l][i] + a_ij[4][3]*q_fields_RK4[which_q_field]->phi[k][l][i]+a_ij[4][4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->chi[k][l][i]   = q_fields[which_q_field]->chi[k][l][i] + dt*(a_ij[4][0]*q_fields_RK1[which_q_field]->chi[k][l][i] + a_ij[4][1]*q_fields_RK2[which_q_field]->chi[k][l][i]
                                                                                                                +a_ij[4][2]*q_fields_RK3[which_q_field]->chi[k][l][i] + a_ij[4][3]*q_fields_RK4[which_q_field]->chi[k][l][i]+a_ij[4][4]*q_fields_RK5[which_q_field]->chi[k][l][i]);
                    q_fields_RK_sum[which_q_field]->pi[k][l][i]    = q_fields[which_q_field]->pi[k][l][i]  + dt*(a_ij[4][0]*q_fields_RK1[which_q_field]->pi[k][l][i]  + a_ij[4][1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                +a_ij[4][2]*q_fields_RK3[which_q_field]->pi[k][l][i]  + a_ij[4][3]*q_fields_RK4[which_q_field]->pi[k][l][i] +a_ij[4][4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                }
            }
        }

        metric_RK_sum->A[i]        = metric->A[i]       + dt*(a_ij[4][0]*metric_RK1->A[i]      + a_ij[4][1]*metric_RK2->A[i]        + a_ij[4][2]*metric_RK3->A[i]       + a_ij[4][3]*metric_RK4->A[i]       + a_ij[4][4]*metric_RK5->A[i]);
        metric_RK_sum->B[i]        = metric->B[i]       + dt*(a_ij[4][0]*metric_RK1->B[i]      + a_ij[4][1]*metric_RK2->B[i]        + a_ij[4][2]*metric_RK3->B[i]       + a_ij[4][3]*metric_RK4->B[i]       + a_ij[4][4]*metric_RK5->B[i]);
        metric_RK_sum->D_B[i]      = metric->D_B[i]     + dt*(a_ij[4][0]*metric_RK1->D_B[i]    + a_ij[4][1]*metric_RK2->D_B[i]      + a_ij[4][2]*metric_RK3->D_B[i]     + a_ij[4][3]*metric_RK4->D_B[i]     + a_ij[4][4]*metric_RK5->D_B[i]);
        metric_RK_sum->U_tilda[i]  = metric->U_tilda[i] + dt*(a_ij[4][0]*metric_RK1->U_tilda[i]+ a_ij[4][1]*metric_RK2->U_tilda[i]  + a_ij[4][2]*metric_RK3->U_tilda[i] + a_ij[4][3]*metric_RK4->U_tilda[i] + a_ij[4][4]*metric_RK5->U_tilda[i]);
        metric_RK_sum->K[i]        = metric->K[i]       + dt*(a_ij[4][0]*metric_RK1->K[i]      + a_ij[4][1]*metric_RK2->K[i]        + a_ij[4][2]*metric_RK3->K[i]       + a_ij[4][3]*metric_RK4->K[i]       + a_ij[4][4]*metric_RK5->K[i]);
        metric_RK_sum->K_B[i]      = metric->K_B[i]     + dt*(a_ij[4][0]*metric_RK1->K_B[i]    + a_ij[4][1]*metric_RK2->K_B[i]      + a_ij[4][2]*metric_RK3->K_B[i]     + a_ij[4][3]*metric_RK4->K_B[i]     + a_ij[4][4]*metric_RK5->K_B[i]);
        metric_RK_sum->lambda[i]   = metric->lambda[i]  + dt*(a_ij[4][0]*metric_RK1->lambda[i] + a_ij[4][1]*metric_RK2->lambda[i]   + a_ij[4][2]*metric_RK3->lambda[i]  + a_ij[4][3]*metric_RK4->lambda[i]  + a_ij[4][4]*metric_RK5->lambda[i]);
        metric_RK_sum->alpha[i]    = metric->alpha[i]   + dt*(a_ij[4][0]*metric_RK1->alpha[i]  + a_ij[4][1]*metric_RK2->alpha[i]    + a_ij[4][2]*metric_RK3->alpha[i]   + a_ij[4][3]*metric_RK4->alpha[i]   + a_ij[4][4]*metric_RK5->alpha[i]);
        metric_RK_sum->D_alpha[i]  = metric->D_alpha[i] + dt*(a_ij[4][0]*metric_RK1->D_alpha[i]+ a_ij[4][1]*metric_RK2->D_alpha[i]  + a_ij[4][2]*metric_RK3->D_alpha[i] + a_ij[4][3]*metric_RK4->D_alpha[i] + a_ij[4][4]*metric_RK5->D_alpha[i]);

    }
    df_dt(cos_const, alpha_AB_pi_mode, c_fields_RK_sum, c_fields_RK5, q_fields_RK_sum, q_fields_RK5, metric_RK_sum, metric_RK5);

}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions that save the variable fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_points(){
    double r[lattice_size_buff];
    make_points(r);
    FILE * pointsout;
    pointsout=fopen("points10.txt", "w");
    for (int i=0;i<lattice_size_buff;++i){
        fprintf(pointsout,"%.20f ",r[i]);
    }
    fclose(pointsout);
}

void save_field_t(double field[evolve_time_int_per_five][lattice_size_buff]) {

    FILE* finout;
    finout = fopen("r_mat.txt", "w");
    for (int n = 0; n < evolve_time_int_per_five; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", field[n][m]);
        }
    }
    fclose(finout);
}

void save_field_t_q(double field[evolve_time_int_per_five][lattice_size_buff]) {

    FILE* finout;
    finout = fopen("r_mat_q.txt", "w");
    for (int n = 0; n < evolve_time_int_per_five; n++) {
        fprintf(finout, "\n");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", field[n][m]);
        }
    }
    fclose(finout);
}

void save_A(double *field){

    FILE * finout;
    finout=fopen("A.txt", "w");
    for (int m=0;m<lattice_size_buff;++m){
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_B(double *field){

    FILE * finout;
    finout=fopen("B.txt", "w");
    for (int m=0;m<lattice_size_buff;++m){
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_K(double *field){

    FILE * finout;
    finout=fopen("K.txt", "w");
    for (int m=0;m<lattice_size_buff;++m){
        fprintf(finout, "%.20f ",field[m]);
    }
    fclose(finout);
}
void save_K_B(double *field){

    FILE * finout;
    finout=fopen("K_B.txt", "w");
    for (int m=0;m<lattice_size_buff;++m){
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_lambda(double* field) {

    FILE* finout;
    finout = fopen("lambda.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_alpha_r(double* field) {

    FILE* finout;
    finout = fopen("alpha_r.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_D_B(double* field) {

    FILE* finout;
    finout = fopen("D_B.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_U_tilda(double* field) {

    FILE* finout;
    finout = fopen("U_tilda.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_ham_r(double* field) {

    FILE* finout;
    finout = fopen("ham_r3.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_mass_MS(double* field) {

    FILE* finout;
    finout = fopen("MS_mass.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_mass_MS_AH(double* field) {

    FILE* finout;
    finout = fopen("MS_mass_AH.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_mass_r(double* field) {

    FILE* finout;
    finout = fopen("BH_mass.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_D_alpha(double* field) {

    FILE* finout;
    finout = fopen("D_alpha.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_field_comp(__complex__ double *field){

    FILE * finout;
    finout=fopen("quantum_mode.txt", "w");
    for (int m=0;m<lattice_size_buff;++m){
        fprintf(finout, "%.150f ",__real__ field[m]);
    }
    fclose(finout);
}
void save_field_comp1(__complex__ double *field){

    FILE * finout;
    finout=fopen("data_comp_1_f.txt", "w");
    for (int m=0;m<lattice_size_buff;++m){
        fprintf(finout, "%.150f ",__real__ field[m]);
    }
    fclose(finout);
}

void save_horizon_i(double *field){

    FILE * finout;
    finout=fopen("horizon_i.txt", "w");
    for (int m=0;m<evolve_time_int;++m){
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_horizon_r(double* field) {

    FILE* finout;
    finout = fopen("horizon_r55s.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}

void save_phi1(double* field) {
    FILE* finout;
    finout = fopen("phi1.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_phi2(double* field) {

    FILE* finout;
    finout = fopen("phi2.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_phi3(double* field) {

    FILE* finout;
    finout = fopen("phi3.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_rho(double* field) {

    FILE* finout;
    finout = fopen("rho.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_ricci(double* field) {

    FILE* finout;
    finout = fopen("ricci.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}

void save_Aini(double* field) {
        FILE * finout;
        finout=fopen("A_ini.txt", "w");
        for (int m=0;m<lattice_size_buff;++m){
            fprintf(finout, "%.20f ", field[m]);
        }
        fclose(finout);
}
void save_alpha(double* field) {

    FILE* finout;
    finout = fopen("alpha_r0.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_mass(double* field) {

    FILE* finout;
    finout = fopen("adm.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}
void save_ham(double* field) {

    FILE* finout;
    finout = fopen("ham.txt", "w");
    for (int m = 0; m < evolve_time_int; ++m) {
        fprintf(finout, "%.20f ", field[m]);
    }
    fclose(finout);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that checks the hamiltonian constraint */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void hamiltonian_constraint(int n, double cos_const, Classical_fields* c_fields, Quantum_fields** q_fields,Metric_Fields *metric, double ham[evolve_time_int]){
    
    double r[lattice_size_buff];
    make_points(r);
    double rho[lattice_size_buff];
    double lambda_B_over_A[lattice_size_buff];

    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for (int i = 0; i < lattice_size_buff; i++) {

        double A, B;
        double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        Bi_Linears    bi_linears;


        A = metric->A[i];
        B = metric->B[i];

        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric);

        phi_phi = bi_linears.phi_phi;
        chi_chi = bi_linears.chi_chi;
        pi_pi = bi_linears.pi_pi;
        chi_pi = bi_linears.chi_pi;
        del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

        //rho[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq - cos_const;
        rho[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi);

        lambda_B_over_A[i] = metric->lambda[i] * B / A;
    }
    for (int i=buff_size; i<lattice_size/2; i++){
        ham[n] = ham[n] + pow((first_deriv(i, metric->D_B)
                + (r[i]!=0.0 ? 1/r[i]*(metric->lambda[i] + metric->D_B[i] - metric->U_tilda[i] - 4.0*lambda_B_over_A[i]) 
                                                                                  : first_deriv(i, metric->lambda) + first_deriv(i, metric->D_B)- first_deriv(i, metric->U_tilda)-4.0*first_deriv(i, lambda_B_over_A))
                - metric->D_B[i]*(0.25* metric->D_B[i] + 0.5* metric->U_tilda[i] + 2.0 * lambda_B_over_A[i])
                -metric->A[i]*metric->K_B[i]*(2.0* metric->K[i]-3.0* metric->K_B[i])
                + metric->A[i]*(rho[i])*M_P*M_P), 2);
        
    }
    ham[n] = (ham[n])/lattice_size;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves all the stress tensor components at various stages of the evolution */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_stress_tensor(int n, double cos_const, Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double r[lattice_size_buff];
    make_points(r);
    double rho[lattice_size_buff], j_A[lattice_size_buff], S_A[lattice_size_buff], S_B[lattice_size_buff];
    
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for (int i=0; i<lattice_size_buff; i++){

        double A, B;
        double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        Bi_Linears    bi_linears;


        A   = metric->A[i];
        B   = metric->B[i];

        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric);

        phi_phi = bi_linears.phi_phi;// -c_fields->phi[i] * c_fields->phi[i];
        chi_chi = bi_linears.chi_chi;// -c_fields->chi[i] * c_fields->chi[i];
        pi_pi = bi_linears.pi_pi;//   -c_fields->pi[i] * c_fields->pi[i];
        chi_pi = bi_linears.chi_pi;//  -c_fields->chi[i] * c_fields->pi[i];
        del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

        rho[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq - cos_const;
        j_A[i] =-chi_pi / (sqrt(A) * B);
        S_A[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq + cos_const;
        S_B[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) - chi_chi) + cos_const;


    }
    // save ham constr
    // saving stress-energy tensor for different time steps
    if (n == 0) {
        FILE* finout;
        finout = fopen("rho(0).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(0).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(0).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(0).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 1) {
        FILE* finout;
        finout = fopen("rho(50).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(50).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(50).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(50).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 2) {
        FILE* finout;
        finout = fopen("rho(100).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(100).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(100).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(100).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 3) {
        FILE* finout;
        finout = fopen("rho(150).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(150).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(150).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(150).txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 4) {
    FILE* finout;
    finout = fopen("rho(200).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(200).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(200).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(200).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 5) {
    FILE* finout;
    finout = fopen("rho(300).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(300).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(300).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(300).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 6) {
    FILE* finout;
    finout = fopen("rho(400).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(400).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(400).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(400).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 7) {
    FILE* finout;
    finout = fopen("rho(500).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(500).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(500).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(500).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 8) {
    FILE* finout;
    finout = fopen("rho(750).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(750).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(750).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(750).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 9) {
    FILE* finout;
    finout = fopen("rho(1000).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(1000).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(1000).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(1000).txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", S_B[m]);
    }
    fclose(finout3);
    }
    // else if (n == 10) {
    // FILE* finout;
    // finout = fopen("rho(1200).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout, "%.40f ", rho[m]);
    // }
    // fclose(finout);

    // FILE* finout1;
    // finout1 = fopen("j_A(1200).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout1, "%.40f ", j_A[m]);
    // }
    // fclose(finout1);

    // FILE* finout2;
    // finout2 = fopen("S_A(1200).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout2, "%.40f ", S_A[m]);
    // }
    // fclose(finout2);

    // FILE* finout3;
    // finout3 = fopen("S_B(1200).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout3, "%.40f ", S_B[m]);
    // }
    // fclose(finout3);
    // }
    // else if (n == 11) {
    // FILE* finout;
    // finout = fopen("rho(1500).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout, "%.40f ", rho[m]);
    // }
    // fclose(finout);

    // FILE* finout1;
    // finout1 = fopen("j_A(1500).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout1, "%.40f ", j_A[m]);
    // }
    // fclose(finout1);

    // FILE* finout2;
    // finout2 = fopen("S_A(1500).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout2, "%.40f ", S_A[m]);
    // }
    // fclose(finout2);

    // FILE* finout3;
    // finout3 = fopen("S_B(1500).txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout3, "%.40f ", S_B[m]);
    // }
    // fclose(finout3);
    // }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves all the stress tensor components at various stages of the evolution */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_stress_tensor_q(int n, double cos_const, Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){
    double r[lattice_size_buff];
    make_points(r);
    double q_rho[lattice_size_buff], q_j_A[lattice_size_buff], q_S_A[lattice_size_buff], q_S_B[lattice_size_buff];
    
    #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
    for (int i=0; i<lattice_size_buff; i++){

        double A, B;
        double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        Bi_Linears    bi_linears;


        A   = metric->A[i];
        B   = metric->B[i];

        set_bi_linears_q(i, &bi_linears, c_fields, q_fields, metric);

        phi_phi = bi_linears.phi_phi;// -c_fields->phi[i] * c_fields->phi[i];
        chi_chi = bi_linears.chi_chi;// -c_fields->chi[i] * c_fields->chi[i];
        pi_pi = bi_linears.pi_pi;//   -c_fields->pi[i] * c_fields->pi[i];
        chi_pi = bi_linears.chi_pi;//  -c_fields->chi[i] * c_fields->pi[i];
        del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

        q_rho[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq - cos_const;
        q_j_A[i] =-chi_pi / (sqrt(A) * B);
        q_S_A[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) - 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq + cos_const;
        q_S_B[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) - chi_chi) + cos_const;


    }
    // save ham constr
    // saving stress-energy tensor for different time steps
    if (n == 0) {
        FILE* finout;
        finout = fopen("rho(0)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", q_rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(0)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", q_j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(0)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", q_S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(0)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", q_S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 1) {
        FILE* finout;
        finout = fopen("rho(50)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", q_rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(50)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", q_j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(50)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", q_S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(50)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", q_S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 2) {
        FILE* finout;
        finout = fopen("rho(100)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", q_rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(100)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", q_j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(100)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", q_S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(100)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", q_S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 3) {
        FILE* finout;
        finout = fopen("rho(150)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout, "%.40f ", q_rho[m]);
        }
        fclose(finout);

        FILE* finout1;
        finout1 = fopen("j_A(150)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout1, "%.40f ", q_j_A[m]);
        }
        fclose(finout1);

        FILE* finout2;
        finout2 = fopen("S_A(150)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout2, "%.40f ", q_S_A[m]);
        }
        fclose(finout2);

        FILE* finout3;
        finout3 = fopen("S_B(150)q.txt", "w");
        for (int m = 0; m < lattice_size_buff; ++m) {
            fprintf(finout3, "%.40f ", q_S_B[m]);
        }
        fclose(finout3);
    }
    else if (n == 4) {
    FILE* finout;
    finout = fopen("rho(200)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(200)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(200)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(200)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 5) {
    FILE* finout;
    finout = fopen("rho(300)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(300)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(300)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(300)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 6) {
    FILE* finout;
    finout = fopen("rho(400)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(400)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(400)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(400)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 7) {
    FILE* finout;
    finout = fopen("rho(500)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(500)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(500)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(500)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 8) {
    FILE* finout;
    finout = fopen("rho(600)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(600)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(600)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(600)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 9) {
    FILE* finout;
    finout = fopen("rho(700)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(700)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(700)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(700)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 10) {
    FILE* finout;
    finout = fopen("rho(800)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(800)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(800)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(800)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 11) {
    FILE* finout;
    finout = fopen("rho(900)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(900)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(900)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(900)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    else if (n == 12) {
    FILE* finout;
    finout = fopen("rho(1000)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout, "%.40f ", q_rho[m]);
    }
    fclose(finout);

    FILE* finout1;
    finout1 = fopen("j_A(1000)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout1, "%.40f ", q_j_A[m]);
    }
    fclose(finout1);

    FILE* finout2;
    finout2 = fopen("S_A(1000)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout2, "%.40f ", q_S_A[m]);
    }
    fclose(finout2);

    FILE* finout3;
    finout3 = fopen("S_B(1000)q.txt", "w");
    for (int m = 0; m < lattice_size_buff; ++m) {
        fprintf(finout3, "%.40f ", q_S_B[m]);
    }
    fclose(finout3);
    }
    // else if (n == 10) {
    // FILE* finout;
    // finout = fopen("rho(1200)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout, "%.40f ", q_rho[m]);
    // }
    // fclose(finout);

    // FILE* finout1;
    // finout1 = fopen("j_A(1200)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout1, "%.40f ", q_j_A[m]);
    // }
    // fclose(finout1);

    // FILE* finout2;
    // finout2 = fopen("S_A(1200)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout2, "%.40f ", q_S_A[m]);
    // }
    // fclose(finout2);

    // FILE* finout3;
    // finout3 = fopen("S_B(1200)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout3, "%.40f ", q_S_B[m]);
    // }
    // fclose(finout3);
    // }
    // else if (n == 11) {
    // FILE* finout;
    // finout = fopen("rho(1500)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout, "%.40f ", q_rho[m]);
    // }
    // fclose(finout);

    // FILE* finout1;
    // finout1 = fopen("j_A(1500)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout1, "%.40f ", q_j_A[m]);
    // }
    // fclose(finout1);

    // FILE* finout2;
    // finout2 = fopen("S_A(1500)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout2, "%.40f ", q_S_A[m]);
    // }
    // fclose(finout2);

    // FILE* finout3;
    // finout3 = fopen("S_B(1500)q.txt", "w");
    // for (int m = 0; m < lattice_size_buff; ++m) {
    //     fprintf(finout3, "%.40f ", q_S_B[m]);
    // }
    // fclose(finout3);
    // }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that saves correlation function at the various stages of the simulation from r=0 to r=r_max */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void save_correlator_half(int n, Classical_fields *c_fields, Quantum_fields** q_fields, double** correlator_half, double** correlator_half_pp) {

    //double correlator_half[lattice_size][lattice_size];

    #pragma omp parallel for num_threads(threads_number)
    for (int i = buff_size; i < lattice_size_buff; ++i) {
        for (int j = buff_size; j < lattice_size_buff; ++j) {
            __complex__ double Phi_mode1, Phi_mode2, Pi_mode1, Pi_mode2, Chi_mode1, Chi_mode2;
            double phi_phi, pi_pi, chi_chi;
            double r1, r2, r_l1, r_l2;
            int l_value;
            phi_phi = 0;
            pi_pi = 0;
            chi_chi = 0;

            r1=(i-buff_size)*dr;
            r2=(j-buff_size)*dr;
            //r1 = r_star_fin[i - buff_size];
            //r2 = r_star_fin[j - buff_size];
            // phi_phi = c_fields->q_pi[i]*c_fields->q_pi[j];

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                        l_value = l_start + l * l_step;
                        r_l1 = pow(r1, l_value);
                        r_l2 = pow(r2, l_value);

                        Phi_mode1 = r_l1 * (q_fields[which_q_field]->phi[k][l][i]);

                        Phi_mode2 = r_l2 * (q_fields[which_q_field]->phi[k][l][j]);

                        phi_phi = phi_phi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * __real__(Phi_mode1 * conj(Phi_mode2));

                        // Chi_mode1 = r_l1 * (q_fields[which_q_field]->chi[k][l][i]);

                        // Chi_mode2 = r_l2 * (q_fields[which_q_field]->chi[k][l][j]);

                        // chi_chi = chi_chi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * __real__(Chi_mode1 * conj(Chi_mode2));
                    }
                }
            }
            correlator_half[i - buff_size][j - buff_size] = phi_phi;// -c_fields->phi[i] * c_fields->phi[j];

            for (int k = 0; k < number_of_k_modes; ++k) {
                for (int l = 0; l < number_of_l_modes; ++l) {
                    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
                        l_value = l_start + l * l_step;
                        r_l1 = pow(r1, l_value);
                        r_l2 = pow(r2, l_value);

                        Pi_mode1 = r_l1 * (q_fields[which_q_field]->pi[k][l][i]);

                        Pi_mode2 = r_l2 * (q_fields[which_q_field]->pi[k][l][j]);


                        pi_pi = pi_pi + ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * __real__(Pi_mode1 * conj(Pi_mode2));

                        //phi1_phi1 = phi1_phi1 + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * __real__ (Phi_mode1 * conj(Phi_mode1));
                        //phi2_phi2 = phi2_phi2 + hbar * ghost_or_physical[which_q_field] * dk / (4.0 * PI) * (2.0 * l_value + 1.0) * __real__ (Phi_mode2 * conj(Phi_mode2));
                    }
                }
            }
            correlator_half_pp[i - buff_size][j - buff_size] = pi_pi;
            // correlator_half_cc[i - buff_size][j - buff_size] = chi_chi;
        }
    }
        if (n == 0) {
        FILE* finout;
        finout = fopen("phi_phi_n=0.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=0.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=0.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 1) {
        FILE* finout;
        finout = fopen("phi_phi_n=550.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=550.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=10.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 2) {
        FILE* finout;
        finout = fopen("phi_phi_n=650.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=650.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=25.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 3) {
        FILE* finout;
        finout = fopen("phi_phi_n=50.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=50.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=50.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 4) {
        FILE* finout;
        finout = fopen("phi_phi_n=75.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=75.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=75.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 5) {
        FILE* finout;
        finout = fopen("phi_phi_n=100.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=100.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=100.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 6) {
        FILE* finout;
        finout = fopen("phi_phi_n=125.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=125.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=125.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 7) {
        FILE* finout;
        finout = fopen("phi_phi_n=150.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=150.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=150.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 8) {
        FILE* finout;
        finout = fopen("phi_phi_n=175.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=175.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=175.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 9) {
        FILE* finout;
        finout = fopen("phi_phi_n=200.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=200.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=200.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 10) {
        FILE* finout;
        finout = fopen("phi_phi_n=250.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=250.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=250.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 11) {
        FILE* finout;
        finout = fopen("phi_phi_n=300.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=300.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=300.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 12) {
        FILE* finout;
        finout = fopen("phi_phi_n=350.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=350.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=350.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 13) {
        FILE* finout;
        finout = fopen("phi_phi_n=400.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=400.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=400.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 14) {
        FILE* finout;
        finout = fopen("phi_phi_n=450.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=450.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=450.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 15) {
        FILE* finout;
        finout = fopen("phi_phi_n=500.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=500.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=500.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 16) {
        FILE* finout;
        finout = fopen("phi_phi_n=600.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=600.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=600.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 17) {
        FILE* finout;
        finout = fopen("phi_phi_n=700.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=700.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=700.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 18) {
        FILE* finout;
        finout = fopen("phi_phi_n=800.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=800.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=800.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 19) {
        FILE* finout;
        finout = fopen("phi_phi_n=900.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=900.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=900.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    else if (n == 20) {
        FILE* finout;
        finout = fopen("phi_phi_n=1000.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout, "%.20f ", (correlator_half[i][j]));
            }
        }
        fclose(finout);
        
        FILE* finout1;
        finout1 = fopen("pi_pi_n=1000.txt", "w");
        for (int i = 0; i < lattice_size; ++i) {
            fprintf(finout1, "\n");
            for (int j = 0; j < lattice_size; ++j) {
                fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
            }
        }
        fclose(finout1);
        
        // FILE* finout2;
        // finout2 = fopen("chi_chi_n=1000.txt", "w");
        // for (int i = 0; i < lattice_size; ++i) {
        //     fprintf(finout2, "\n");
        //     for (int j = 0; j < lattice_size; ++j) {
        //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
        //     }
        // }
        // fclose(finout2);
    }
    // else if (n == 21) {
    //     FILE* finout;
    //     finout = fopen("phi_phi_n=1100.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
        
    //     FILE* finout1;
    //     finout1 = fopen("pi_pi_n=1100.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout1, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
    //         }
    //     }
    //     fclose(finout1);
        
    //     // FILE* finout2;
    //     // finout2 = fopen("chi_chi_n=1000.txt", "w");
    //     // for (int i = 0; i < lattice_size; ++i) {
    //     //     fprintf(finout2, "\n");
    //     //     for (int j = 0; j < lattice_size; ++j) {
    //     //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
    //     //     }
    //     // }
    //     // fclose(finout2);
    // }
    // else if (n == 22) {
    //     FILE* finout;
    //     finout = fopen("phi_phi_n=1200.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
        
    //     FILE* finout1;
    //     finout1 = fopen("pi_pi_n=1200.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout1, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
    //         }
    //     }
    //     fclose(finout1);
        
    //     // FILE* finout2;
    //     // finout2 = fopen("chi_chi_n=1000.txt", "w");
    //     // for (int i = 0; i < lattice_size; ++i) {
    //     //     fprintf(finout2, "\n");
    //     //     for (int j = 0; j < lattice_size; ++j) {
    //     //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
    //     //     }
    //     // }
    //     // fclose(finout2);
    // }
    // else if (n == 23) {
    //     FILE* finout;
    //     finout = fopen("phi_phi_n=1300.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
        
    //     FILE* finout1;
    //     finout1 = fopen("pi_pi_n=1300.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout1, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
    //         }
    //     }
    //     fclose(finout1);
        
    //     // FILE* finout2;
    //     // finout2 = fopen("chi_chi_n=1000.txt", "w");
    //     // for (int i = 0; i < lattice_size; ++i) {
    //     //     fprintf(finout2, "\n");
    //     //     for (int j = 0; j < lattice_size; ++j) {
    //     //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
    //     //     }
    //     // }
    //     // fclose(finout2);
    // }
    // else if (n == 24) {
    //     FILE* finout;
    //     finout = fopen("phi_phi_n=1400.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
        
    //     FILE* finout1;
    //     finout1 = fopen("pi_pi_n=1400.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout1, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
    //         }
    //     }
    //     fclose(finout1);
        
    //     // FILE* finout2;
    //     // finout2 = fopen("chi_chi_n=1000.txt", "w");
    //     // for (int i = 0; i < lattice_size; ++i) {
    //     //     fprintf(finout2, "\n");
    //     //     for (int j = 0; j < lattice_size; ++j) {
    //     //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
    //     //     }
    //     // }
    //     // fclose(finout2);
    // }
    // else if (n == 25) {
    //     FILE* finout;
    //     finout = fopen("phi_phi_n=1500.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
        
    //     FILE* finout1;
    //     finout1 = fopen("pi_pi_n=1500.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout1, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout1, "%.20f ", (correlator_half_pp[i][j]));
    //         }
    //     }
    //     fclose(finout1);
        
    //     // FILE* finout2;
    //     // finout2 = fopen("chi_chi_n=1000.txt", "w");
    //     // for (int i = 0; i < lattice_size; ++i) {
    //     //     fprintf(finout2, "\n");
    //     //     for (int j = 0; j < lattice_size; ++j) {
    //     //         fprintf(finout2, "%.20f ", (correlator_half_cc[i][j]));
    //     //     }
    //     // }
    //     // fclose(finout2);
    // }
    // if (n == 0) {
    //     FILE* finout;
    //     finout = fopen("n=0.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 1) {
    //     FILE* finout;
    //     finout = fopen("n=10.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 2) {
    //     FILE* finout;
    //     finout = fopen("n=25.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 3) {
    //     FILE* finout;
    //     finout = fopen("n=50.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 4) {
    //     FILE* finout;
    //     finout = fopen("n=75.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 5) {
    //     FILE* finout;
    //     finout = fopen("n=100.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 6) {
    //     FILE* finout;
    //     finout = fopen("n=125.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 7) {
    //     FILE* finout;
    //     finout = fopen("n=150.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 8) {
    //     FILE* finout;
    //     finout = fopen("n=175.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 9) {
    //     FILE* finout;
    //     finout = fopen("n=200.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 10) {
    //     FILE* finout;
    //     finout = fopen("n=250.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 11) {
    //     FILE* finout;
    //     finout = fopen("n=300.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 12) {
    //     FILE* finout;
    //     finout = fopen("n=350.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 13) {
    //     FILE* finout;
    //     finout = fopen("n=400.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 14) {
    //     FILE* finout;
    //     finout = fopen("n=450.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 15) {
    //     FILE* finout;
    //     finout = fopen("n=500.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 16) {
    //     FILE* finout;
    //     finout = fopen("n=600.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 17) {
    //     FILE* finout;
    //     finout = fopen("n=700.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 18) {
    //     FILE* finout;
    //     finout = fopen("n=800.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 19) {
    //     FILE* finout;
    //     finout = fopen("n=900.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
    // else if (n == 20) {
    //     FILE* finout;
    //     finout = fopen("n=1000.txt", "w");
    //     for (int i = 0; i < lattice_size; ++i) {
    //         fprintf(finout, "\n");
    //         for (int j = 0; j < lattice_size; ++j) {
    //             fprintf(finout, "%.20f ", (correlator_half[i][j]));
    //         }
    //     }
    //     fclose(finout);
    // }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Full evolution function */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void fullEvolution(double b_i[nu_legendre], double a_ij[nu_legendre][nu_legendre], __complex__ double ****alpha_AB_pi_mode, Classical_fields *c_fields_RK1, Classical_fields *c_fields_RK2, Classical_fields *c_fields_RK3,
                                    Classical_fields *c_fields_RK4, Classical_fields *c_fields_RK5, Classical_fields *c_fields_RK_sum, Classical_fields *c_fields,
                                    Quantum_fields **q_fields_RK1, Quantum_fields **q_fields_RK2, Quantum_fields **q_fields_RK3, Quantum_fields **q_fields_RK4,
                                    Quantum_fields **q_fields_RK5, Quantum_fields **q_fields_RK_sum, Quantum_fields **q_fields,
                                    Metric_Fields *metric_RK1, Metric_Fields *metric_RK2, Metric_Fields *metric_RK3, Metric_Fields *metric_RK4,
                                    Metric_Fields *metric_RK5, Metric_Fields *metric_RK_sum, Metric_Fields *metric, 
                                    double alpha_save[evolve_time_int], double cos_const, double timestart,
                                    Quantum_fields** q_fields_t_star, double** correlator_half, double** correlator_half_pp, double field_save[evolve_time_int_per_five][lattice_size_buff],
                                    double field_save_q[evolve_time_int_per_five][lattice_size_buff]){
    double r[lattice_size_buff];
    make_points(r);

    double apparent_horizon_r[evolve_time_int];
    double apparent_horizon_i[evolve_time_int];
    double mass_ADM[evolve_time_int];
    double mass_BH[evolve_time_int];
    double mass_MS[evolve_time_int];
    double mass_MS_AH[evolve_time_int];
    double ham[evolve_time_int];

    int k = 0;
    double T = 0;

    

    for (int n = 0; n<evolve_time_int; ++n){ //give the u and pi a starting point for the iterations involved in finding the implicit evolution step
            apparent_horizon_i[n] = 0.0;
            apparent_horizon_r[n] = 0.0;
            mass_BH[n] = 0.0;
            mass_MS_AH[n] = 0.0;
            mass_MS[n] = 0.0;
            mass_ADM[n] = 0.0;
            ham[n] = 0.0;

            hamiltonian_constraint(n, cos_const, c_fields, q_fields, metric, ham);

            #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
            for(int i=0; i<lattice_size_buff; ++i){
                c_fields_RK1->phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK2->phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK3->phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK4->phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK5->phi[i]  = 0.0;//fields->phi[i];

                c_fields_RK1->pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK2->pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK3->pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK4->pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK5->pi[i]  = 0.0;//fields->pi[i];

                c_fields_RK1->chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK2->chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK3->chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK4->chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK5->chi[i]  = 0.0;//fields->chi[i];

                c_fields_RK1->q_phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK2->q_phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK3->q_phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK4->q_phi[i]  = 0.0;//fields->phi[i];
                c_fields_RK5->q_phi[i]  = 0.0;//fields->phi[i];

                c_fields_RK1->q_pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK2->q_pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK3->q_pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK4->q_pi[i]  = 0.0;//fields->pi[i];
                c_fields_RK5->q_pi[i]  = 0.0;//fields->pi[i];

                c_fields_RK1->q_chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK2->q_chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK3->q_chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK4->q_chi[i]  = 0.0;//fields->chi[i];
                c_fields_RK5->q_chi[i]  = 0.0;//fields->chi[i];


                for(int l=0; l<number_of_l_modes; ++l){
                    //printf("i=%d, k=%d and process: %d\n",i, k, omp_get_thread_num());
                    for(int k=0; k<number_of_k_modes; ++k){
                        for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                            q_fields_RK1[which_q_field]->phi[k][l][i]   = 0.0;
                            q_fields_RK2[which_q_field]->phi[k][l][i]   = 0.0;
                            q_fields_RK3[which_q_field]->phi[k][l][i]   = 0.0;
                            q_fields_RK4[which_q_field]->phi[k][l][i]   = 0.0;
                            q_fields_RK5[which_q_field]->phi[k][l][i]   = 0.0;

                            q_fields_RK1[which_q_field]->chi[k][l][i]   = 0.0;
                            q_fields_RK2[which_q_field]->chi[k][l][i]   = 0.0;
                            q_fields_RK3[which_q_field]->chi[k][l][i]   = 0.0;
                            q_fields_RK4[which_q_field]->chi[k][l][i]   = 0.0;
                            q_fields_RK5[which_q_field]->chi[k][l][i]   = 0.0;

                            q_fields_RK1[which_q_field]->pi[k][l][i]    = 0.0;
                            q_fields_RK2[which_q_field]->pi[k][l][i]    = 0.0;
                            q_fields_RK3[which_q_field]->pi[k][l][i]    = 0.0;
                            q_fields_RK4[which_q_field]->pi[k][l][i]    = 0.0;
                            q_fields_RK5[which_q_field]->pi[k][l][i]    = 0.0;
                        }
                    }
                }

                metric_RK1->A[i]    = 0.0;//metric->A[i];
                metric_RK2->A[i]    = 0.0;//metric->A[i];
                metric_RK3->A[i]    = 0.0;//metric->A[i];
                metric_RK4->A[i]    = 0.0;//metric->A[i];
                metric_RK5->A[i]    = 0.0;//metric->A[i];

                metric_RK1->B[i]    = 0.0;//metric->B[i];
                metric_RK2->B[i]    = 0.0;//metric->B[i];
                metric_RK3->B[i]    = 0.0;//metric->B[i];
                metric_RK4->B[i]    = 0.0;//metric->B[i];
                metric_RK5->B[i]    = 0.0;//metric->B[i];

                metric_RK1->D_B[i]    = 0.0;//metric->D_B[i];
                metric_RK2->D_B[i]    = 0.0;//metric->D_B[i];
                metric_RK3->D_B[i]    = 0.0;//metric->D_B[i];
                metric_RK4->D_B[i]    = 0.0;//metric->D_B[i];
                metric_RK5->D_B[i]    = 0.0;//metric->D_B[i];

                metric_RK1->U_tilda[i]    = 0.0;//metric->U_tilda[i];
                metric_RK2->U_tilda[i]    = 0.0;//metric->U_tilda[i];
                metric_RK3->U_tilda[i]    = 0.0;//metric->U_tilda[i];
                metric_RK4->U_tilda[i]    = 0.0;//metric->U_tilda[i];   
                metric_RK5->U_tilda[i]    = 0.0;//metric->U_tilda[i];

                metric_RK1->K[i]    = 0.0;//metric->K[i];
                metric_RK2->K[i]    = 0.0;//metric->K[i];
                metric_RK3->K[i]    = 0.0;//metric->K[i];
                metric_RK4->K[i]    = 0.0;//metric->K[i];
                metric_RK5->K[i]    = 0.0;//metric->K[i];

                metric_RK1->K_B[i]    = 0.0;//metric->K_B[i];
                metric_RK2->K_B[i]    = 0.0;//metric->K_B[i];
                metric_RK3->K_B[i]    = 0.0;//metric->K_B[i];
                metric_RK4->K_B[i]    = 0.0;//metric->K_B[i];
                metric_RK5->K_B[i]    = 0.0;//metric->K_B[i];

                metric_RK1->lambda[i]    = 0.0;//metric->lambda[i];
                metric_RK2->lambda[i]    = 0.0;//metric->lambda[i];
                metric_RK3->lambda[i]    = 0.0;//metric->lambda[i];
                metric_RK4->lambda[i]    = 0.0;//metric->lambda[i];
                metric_RK5->lambda[i]    = 0.0;//metric->lambda[i];

                metric_RK1->alpha[i]    = 0.0;//metric->alpha[i];
                metric_RK2->alpha[i]    = 0.0;//metric->alpha[i];
                metric_RK3->alpha[i]    = 0.0;//metric->alpha[i];
                metric_RK4->alpha[i]    = 0.0;//metric->alpha[i];
                metric_RK5->alpha[i]    = 0.0;//metric->alpha[i];

                metric_RK1->D_alpha[i]    = 0.0;//metric->D_alpha[i];
                metric_RK2->D_alpha[i]    = 0.0;//metric->D_alpha[i];
                metric_RK3->D_alpha[i]    = 0.0;//metric->D_alpha[i];
                metric_RK4->D_alpha[i]    = 0.0;//metric->D_alpha[i];
                metric_RK5->D_alpha[i]    = 0.0;//metric->D_alpha[i];

            }

            //do a few RK iterations in order to converge on the implicit solution
            for(int iter=0;iter<number_of_RK_implicit_iterations;++iter){
                single_RK_convergence_step_RK5(a_ij, alpha_AB_pi_mode, c_fields_RK1, c_fields_RK2, c_fields_RK3, c_fields_RK4, c_fields_RK5, c_fields_RK_sum, c_fields,
                                                     q_fields_RK1, q_fields_RK2, q_fields_RK3, q_fields_RK4, q_fields_RK5, q_fields_RK_sum, q_fields,
                                                    metric_RK1,    metric_RK2, metric_RK3, metric_RK4, metric_RK5, metric_RK_sum, metric, cos_const);
            }
            //add up the RK contributions

            #pragma omp parallel for num_threads(threads_number)//num_threads(lattice_size_buff)
            for(int i=0; i<lattice_size_buff; ++i){
                c_fields->phi[i]= c_fields->phi[i]+ dt*(b_i[0]*c_fields_RK1->phi[i]+b_i[1]*c_fields_RK2->phi[i]+b_i[2]*c_fields_RK3->phi[i]+b_i[3]*c_fields_RK4->phi[i]+b_i[4]*c_fields_RK5->phi[i]);
                c_fields->pi[i] = c_fields->pi[i] + dt*(b_i[0]*c_fields_RK1->pi[i] +b_i[1]*c_fields_RK2->pi[i] +b_i[2]*c_fields_RK3->pi[i] +b_i[3]*c_fields_RK4->pi[i] +b_i[4]*c_fields_RK5->pi[i]);
                c_fields->chi[i]= c_fields->chi[i]+ dt*(b_i[0]*c_fields_RK1->chi[i]+b_i[1]*c_fields_RK2->chi[i]+b_i[2]*c_fields_RK3->chi[i]+b_i[3]*c_fields_RK4->chi[i]+b_i[4]*c_fields_RK5->chi[i]);

                c_fields->q_phi[i]= c_fields->q_phi[i]+ dt*(b_i[0]*c_fields_RK1->q_phi[i]+b_i[1]*c_fields_RK2->q_phi[i]+b_i[2]*c_fields_RK3->q_phi[i]+b_i[3]*c_fields_RK4->q_phi[i]+b_i[4]*c_fields_RK5->q_phi[i]);
                c_fields->q_pi[i] = c_fields->q_pi[i] + dt*(b_i[0]*c_fields_RK1->q_pi[i] +b_i[1]*c_fields_RK2->q_pi[i] +b_i[2]*c_fields_RK3->q_pi[i] +b_i[3]*c_fields_RK4->q_pi[i] +b_i[4]*c_fields_RK5->q_pi[i]);
                c_fields->q_chi[i]= c_fields->q_chi[i]+ dt*(b_i[0]*c_fields_RK1->q_chi[i]+b_i[1]*c_fields_RK2->q_chi[i]+b_i[2]*c_fields_RK3->q_chi[i]+b_i[3]*c_fields_RK4->q_chi[i]+b_i[4]*c_fields_RK5->q_chi[i]);

                for(int l=0; l<number_of_l_modes; ++l){
                    for(int k=0; k<number_of_k_modes; ++k){
                        for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){

                            q_fields[which_q_field]->phi[k][l][i] = q_fields[which_q_field]->phi[k][l][i] + dt*(b_i[0]*q_fields_RK1[which_q_field]->phi[k][l][i]+b_i[1]*q_fields_RK2[which_q_field]->phi[k][l][i]
                                                                                                                        +b_i[2]*q_fields_RK3[which_q_field]->phi[k][l][i]+b_i[3]*q_fields_RK4[which_q_field]->phi[k][l][i]+b_i[4]*q_fields_RK5[which_q_field]->phi[k][l][i]);
                            q_fields[which_q_field]->chi[k][l][i] = q_fields[which_q_field]->chi[k][l][i] + dt*(b_i[0]*q_fields_RK1[which_q_field]->chi[k][l][i]+b_i[1]*q_fields_RK2[which_q_field]->chi[k][l][i]
                                                                                                                        +b_i[2]*q_fields_RK3[which_q_field]->chi[k][l][i]+b_i[3]*q_fields_RK4[which_q_field]->chi[k][l][i]+b_i[4]*q_fields_RK5[which_q_field]->chi[k][l][i]);
                            q_fields[which_q_field]->pi[k][l][i]  = q_fields[which_q_field]->pi[k][l][i]  + dt*(b_i[0]*q_fields_RK1[which_q_field]->pi[k][l][i] +b_i[1]*q_fields_RK2[which_q_field]->pi[k][l][i]
                                                                                                                        +b_i[2]*q_fields_RK3[which_q_field]->pi[k][l][i] +b_i[3]*q_fields_RK4[which_q_field]->pi[k][l][i] +b_i[4]*q_fields_RK5[which_q_field]->pi[k][l][i]);
                            
                        }
                    }
                }
                metric->A[i]        = metric->A[i]       + dt*(b_i[0]*metric_RK1->A[i]      + b_i[1]*metric_RK2->A[i]        + b_i[2]*metric_RK3->A[i]       + b_i[3]*metric_RK4->A[i]       + b_i[4]*metric_RK5->A[i]);
                metric->B[i]        = metric->B[i]       + dt*(b_i[0]*metric_RK1->B[i]      + b_i[1]*metric_RK2->B[i]        + b_i[2]*metric_RK3->B[i]       + b_i[3]*metric_RK4->B[i]       + b_i[4]*metric_RK5->B[i]);
                metric->D_B[i]      = metric->D_B[i]     + dt*(b_i[0]*metric_RK1->D_B[i]    + b_i[1]*metric_RK2->D_B[i]      + b_i[2]*metric_RK3->D_B[i]     + b_i[3]*metric_RK4->D_B[i]     + b_i[4]*metric_RK5->D_B[i]);
                metric->U_tilda[i]  = metric->U_tilda[i] + dt*(b_i[0]*metric_RK1->U_tilda[i]+ b_i[1]*metric_RK2->U_tilda[i]  + b_i[2]*metric_RK3->U_tilda[i] + b_i[3]*metric_RK4->U_tilda[i] + b_i[4]*metric_RK5->U_tilda[i]);
                metric->K[i]        = metric->K[i]       + dt*(b_i[0]*metric_RK1->K[i]      + b_i[1]*metric_RK2->K[i]        + b_i[2]*metric_RK3->K[i]       + b_i[3]*metric_RK4->K[i]       + b_i[4]*metric_RK5->K[i]);
                metric->K_B[i]      = metric->K_B[i]     + dt*(b_i[0]*metric_RK1->K_B[i]    + b_i[1]*metric_RK2->K_B[i]      + b_i[2]*metric_RK3->K_B[i]     + b_i[3]*metric_RK4->K_B[i]     + b_i[4]*metric_RK5->K_B[i]);
                metric->lambda[i]   = metric->lambda[i]  + dt*(b_i[0]*metric_RK1->lambda[i] + b_i[1]*metric_RK2->lambda[i]   + b_i[2]*metric_RK3->lambda[i]  + b_i[3]*metric_RK4->lambda[i]  + b_i[4]*metric_RK5->lambda[i]);
                metric->alpha[i]    = metric->alpha[i]   + dt*(b_i[0]*metric_RK1->alpha[i]  + b_i[1]*metric_RK2->alpha[i]    + b_i[2]*metric_RK3->alpha[i]   + b_i[3]*metric_RK4->alpha[i]   + b_i[4]*metric_RK5->alpha[i]);
                metric->D_alpha[i]  = metric->D_alpha[i] + dt*(b_i[0]*metric_RK1->D_alpha[i]+ b_i[1]*metric_RK2->D_alpha[i]  + b_i[2]*metric_RK3->D_alpha[i] + b_i[3]*metric_RK4->D_alpha[i] + b_i[4]*metric_RK5->D_alpha[i]);

                
            /* Saving the spatial profile of expectation value of the field (for coherent state, it is c_fields->phi[i]) at each instant of evolution */
            field_save[n][i] = c_fields->phi[i];
            field_save_q[n][i] = c_fields->q_phi[i];

            }
            

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /* Check for apparent horizon */
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            for (int i = buff_size+1; i < lattice_size_buff; i++) {
                double H1, H2, r_AH, B_AH, M_AH;
                H1 = 1.0 / sqrt(metric->A[i-1]) * (2.0 / r[i-1]   + metric->D_B[i-1])  - 2.0 * metric->K_B[i-1];
                H2 = 1.0 / sqrt(metric->A[i])   * (2.0 / r[i]     + metric->D_B[i])    - 2.0 * metric->K_B[i];
                
                if ( H1 < 0 && H2 > 0) {
    
                    r_AH = r[i-1]+dr*(-H1)/(H2-H1);

                    B_AH = (r_AH-r[i-1])/dr*(sqrt(metric->B[i])- sqrt(metric->B[i-1]))+ sqrt(metric->B[i-1]);

                    M_AH = (r[i] * sqrt(metric->B[i])) / 2.0 * (1 + pow(r[i],2) * metric->B[i] * pow(metric->K_B[i],2)-(metric->B[i]/metric->A[i])*(1+r[i]*metric->D_B[i]+pow(r[i],2)*pow(metric->D_B[i],2)/4));

                    apparent_horizon_i[n] = r_AH;
                    apparent_horizon_r[n] = r_AH* B_AH;

                    /* BH mass */
                    mass_BH[n] = r_AH * B_AH/2;
                    printf("horizon is %f\n", r_AH * B_AH);
                    mass_MS_AH[n] = M_AH;
                }

            }


            /* ADM mass */

            mass_ADM[n] =  (r[lattice_size] * sqrt(metric->B[lattice_size])) / 2.0 - (r[lattice_size] * sqrt(metric->B[lattice_size]) * metric->B[lattice_size]) / (8.0 * metric->A[lattice_size]) * pow((r[lattice_size] * metric->D_B[lattice_size]+2),2);

            /* Misner-Sharpe mass */

            mass_MS[n] =  (r[lattice_size] * sqrt(metric->B[lattice_size])) / 2.0 * (1 + pow(r[lattice_size],2) * metric->B[lattice_size] * pow(metric->K_B[lattice_size],2)-(metric->B[lattice_size]/metric->A[lattice_size])*(1+r[lattice_size]*metric->D_B[lattice_size]+pow(r[lattice_size],2)*pow(metric->D_B[lattice_size],2)/4));

            /* Lapse function alpha at r=0 */

            printf("At t=%d alpha at r=0 is alpha=%.20f\n", n, metric->alpha[buff_size]);
            // printf("At t=%d hamiltonian constraint is ham=%10f\n", n, ham);

            alpha_save[n]=metric->alpha[buff_size];

            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /* Save stress tensor and correlators if needed */
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // if (n == 0) {
            //    save_stress_tensor(0, cos_const, c_fields, q_fields, metric);
            // }
            if (n == 50) {
               save_stress_tensor(1, cos_const, c_fields, q_fields, metric);
           }
            if (n == 100) {
               save_stress_tensor(2, cos_const, c_fields, q_fields, metric);
           }
            if (n == 150) {
               save_stress_tensor(3, cos_const, c_fields, q_fields, metric);
            }
            if (n == 200) {
               save_stress_tensor(4, cos_const, c_fields, q_fields, metric);
            }
            if (n == 300) {
               save_stress_tensor(5, cos_const, c_fields, q_fields, metric);
            }
            if (n == 400) {
               save_stress_tensor(6, cos_const, c_fields, q_fields, metric);
            }
            if (n == 500) {
               save_stress_tensor(7, cos_const, c_fields, q_fields, metric);
            }
            if (n == 750) {
               save_stress_tensor(8, cos_const, c_fields, q_fields, metric);
            }
            if (n == 1000) {
               save_stress_tensor(9, cos_const, c_fields, q_fields, metric);
            }
            // if (n == 1200) {
            //    save_stress_tensor(10, cos_const, c_fields, q_fields, metric);
            // }
            // if (n == 1500) {
            //    save_stress_tensor(11, cos_const, c_fields, q_fields, metric);
            // }

            if (n == 50) {
               save_stress_tensor_q(1, cos_const, c_fields, q_fields, metric);
           }
            if (n == 100) {
               save_stress_tensor_q(2, cos_const, c_fields, q_fields, metric);
           }
            if (n == 150) {
               save_stress_tensor_q(3, cos_const, c_fields, q_fields, metric);
            }
            if (n == 200) {
               save_stress_tensor_q(4, cos_const, c_fields, q_fields, metric);
            }
            if (n == 300) {
               save_stress_tensor_q(5, cos_const, c_fields, q_fields, metric);
            }
            if (n == 400) {
               save_stress_tensor_q(6, cos_const, c_fields, q_fields, metric);
            }
            if (n == 500) {
               save_stress_tensor_q(7, cos_const, c_fields, q_fields, metric);
            }
            if (n == 600) {
               save_stress_tensor_q(8, cos_const, c_fields, q_fields, metric);
            }
            if (n == 700) {
               save_stress_tensor_q(9, cos_const, c_fields, q_fields, metric);
            }
            if (n == 800) {
               save_stress_tensor_q(10, cos_const, c_fields, q_fields, metric);
            }
            if (n == 900) {
               save_stress_tensor_q(11, cos_const, c_fields, q_fields, metric);
            }
            if (n == 1000) {
               save_stress_tensor_q(12, cos_const, c_fields, q_fields, metric);
            }
                        
            //if (n == 799) {
            //    save_horizon(apparent_horizon);
            //}
            if (n == 0) {
                save_correlator_half(0, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 550) {
                save_correlator_half(1, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 650) {
                save_correlator_half(2, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 50) {
                save_correlator_half(3, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 75) {
                save_correlator_half(4, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 100) {
                save_correlator_half(5, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 125) {
                save_correlator_half(6, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 150) {
                save_correlator_half(7, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 175) {
                save_correlator_half(8, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 200) {
                save_correlator_half(9, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 250) {
                save_correlator_half(10, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 300) {
                save_correlator_half(11, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 350) {
                save_correlator_half(12, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 400) {
                save_correlator_half(13, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 450) {
                save_correlator_half(14, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 500) {
                save_correlator_half(15, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 600) {
                save_correlator_half(16, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 700) {
                save_correlator_half(17, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 800) {
                save_correlator_half(18, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 900) {
                save_correlator_half(19, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 1000) {
                save_correlator_half(20, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 1100) {
                save_correlator_half(21, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 1200) {
                save_correlator_half(22, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 1300) {
                save_correlator_half(23, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            if (n == 1400) {
                save_correlator_half(24, c_fields, q_fields, correlator_half, correlator_half_pp);
            }
            printf("\n and took %f\n", omp_get_wtime() - timestart);
    }
    save_alpha(alpha_save);
    save_mass(mass_ADM);
    save_mass_MS(mass_MS);
    save_mass_MS_AH(mass_MS_AH);
    save_mass_r(mass_BH);
    save_ham(ham);
    save_horizon_r(apparent_horizon_r);
    save_horizon_i(apparent_horizon_i);
    
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that assigns memory for the fields */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void assign_memory(Classical_fields *fieldd){


    fieldd           = (Classical_fields *)malloc(sizeof(Classical_fields));
    fieldd->phi      = (double*) malloc(lattice_size_buff*sizeof(double));
    fieldd->pi       = (double*) malloc(lattice_size_buff*sizeof(double));
    fieldd->chi      = (double*) malloc(lattice_size_buff*sizeof(double));

    fieldd->q_phi      = (double*) malloc(lattice_size_buff*sizeof(double));
    fieldd->q_pi       = (double*) malloc(lattice_size_buff*sizeof(double));
    fieldd->q_chi      = (double*) malloc(lattice_size_buff*sizeof(double));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Function that frees up the memory */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void free_memory(Classical_fields *c_fields, Quantum_fields **q_fields, Metric_Fields *metric){

    free(metric->A);
    free(metric->B);
    free(metric->D_B);
    free(metric->U_tilda);
    free(metric->K);
    free(metric->K_B);
    free(metric->lambda);
    free(metric->alpha);
    free(metric->D_alpha);
    free(metric);

    free(c_fields->phi);
    free(c_fields->pi);
    free(c_fields->chi);
    free(c_fields->q_phi);
    free(c_fields->q_pi);
    free(c_fields->q_chi);
    free(c_fields);

    for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
        for(int k=0; k<number_of_k_modes; ++k){
            for(int l=0; l<number_of_l_modes; ++l){
                free(q_fields[which_q_field]->phi[k][l]);
                free(q_fields[which_q_field]->chi[k][l]);
                free(q_fields[which_q_field]->pi[k][l]);
            }
            free(q_fields[which_q_field]->phi[k]);
            free(q_fields[which_q_field]->chi[k]);
            free(q_fields[which_q_field]->pi[k]);
        }
        free(q_fields[which_q_field]->phi);
        free(q_fields[which_q_field]->chi);
        free(q_fields[which_q_field]->pi);

        free(q_fields[which_q_field]);
    }
    free(q_fields);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* MAIN */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(){
    double timestart = omp_get_wtime();
    

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* DEFINE VARIABLES AND ASSIGN ALL THE MEMORY *//////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////* Runge-Kutta things *///////////////////////////////////////////////////////////////////////////////////////////////////
    double   a_ij[nu_legendre][nu_legendre];             //coefficients of the Runge-Kutta evolution
    double   b_i[nu_legendre];                           //coefficients of the Runge-Kutta evolution
    double   c_i[nu_legendre];                           //the zeros of P_nu(2c - 1)=0, i.e. P_nu(2c_1 - 1)
    double   GL_matrix_inverse[nu_legendre][nu_legendre];         //this comes from GL_matrix*(a_ij)=(c_i^l) for (a_ij)
                                                                                     //                          (b_i)  (1/l  )     (b_i )

    find_c_i(c_i);
    find_GL_matrix_inverse(c_i, GL_matrix_inverse);
    find_a_ij__b_i(c_i, b_i, a_ij, GL_matrix_inverse);

    /////////////////////* Defining variables and allocating memory *///////////////////////////////////////////////////////////////////////////////////////////////////
    Metric_Fields *metric;
    Metric_Fields *metric_RK1;
    Metric_Fields *metric_RK2;
    Metric_Fields *metric_RK3;
    Metric_Fields *metric_RK4;
    Metric_Fields *metric_RK5;
    Metric_Fields *metric_RK_sum;

    metric           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));

    metric_RK1           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK1->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK1->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));

    metric_RK2           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK2->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK2->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));

    metric_RK3           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK3->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK3->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));

    metric_RK4           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK4->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK4->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));

    metric_RK5           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK5->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK5->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));

    metric_RK_sum           = (Metric_Fields*)malloc(sizeof(Metric_Fields));
    metric_RK_sum->A        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->B        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->D_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->U_tilda  = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->K        = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->K_B      = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->lambda   = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->alpha    = (double*)malloc(lattice_size_buff*sizeof(double));
    metric_RK_sum->D_alpha  = (double*)malloc(lattice_size_buff*sizeof(double));


    Classical_fields *c_fields;
    Classical_fields *c_fields_RK1;
    Classical_fields *c_fields_RK2;
    Classical_fields *c_fields_RK3;
    Classical_fields *c_fields_RK4;
    Classical_fields *c_fields_RK5;
    Classical_fields *c_fields_RK_sum;

    Quantum_fields  **q_fields;
    Quantum_fields  **q_fields_t_star;
    Quantum_fields  **q_fields_RK1;
    Quantum_fields  **q_fields_RK2;
    Quantum_fields  **q_fields_RK3;
    Quantum_fields  **q_fields_RK4;
    Quantum_fields  **q_fields_RK5;
    Quantum_fields  **q_fields_RK_sum;



    //allocate memory for classical fields, accessed with c_fields->phi[i]: Background case and test quantum field
    c_fields      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));

    c_fields_RK1      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields_RK1->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK1->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK1->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK1->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK1->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK1->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));


    c_fields_RK2      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields_RK2->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK2->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK2->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK2->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK2->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK2->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));

    c_fields_RK3      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields_RK3->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK3->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK3->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK3->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK3->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK3->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));

    c_fields_RK4      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields_RK4->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK4->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK4->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK4->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK4->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK4->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));

    c_fields_RK5      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields_RK5->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK5->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK5->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK5->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK5->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK5->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));

    c_fields_RK_sum      = (Classical_fields *)malloc(sizeof(Classical_fields));
    c_fields_RK_sum->phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK_sum->pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK_sum->chi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK_sum->q_phi = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK_sum->q_pi  = (double*) malloc(lattice_size_buff*sizeof(double));
    c_fields_RK_sum->q_chi = (double*) malloc(lattice_size_buff*sizeof(double));
    

    //allocate memory for quantum modes, accessed with q_fields[which_q_field]->phi_mode[k][l][i]
    q_fields            = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_t_star     = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_RK1        = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_RK2        = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_RK3        = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_RK4        = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_RK5        = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));
    q_fields_RK_sum     = (Quantum_fields **)malloc(number_of_q_fields * sizeof(Quantum_fields*));

    __complex__ double**** alpha_AB_pi_mode;
    alpha_AB_pi_mode = (__complex__ double****)malloc(number_of_q_fields * sizeof(__complex__ double***));

    double** correlator_half;
    correlator_half = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        correlator_half[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    double** correlator_half_pp;
    correlator_half_pp = (double**)malloc(lattice_size * sizeof(double*));
    for (int i = 0; i < lattice_size; i++) {
        correlator_half_pp[i] = (double*)malloc(lattice_size * sizeof(double));
    }

    // double** correlator_half_cc;
    // correlator_half = (double**)malloc(lattice_size * sizeof(double*));
    // for (int i = 0; i < lattice_size; i++) {
    //     correlator_half_cc[i] = (double*)malloc(lattice_size * sizeof(double));
    // }

    for(int which_q_field=0;which_q_field<number_of_q_fields;++which_q_field){
        q_fields[which_q_field]         = (Quantum_fields *)malloc(sizeof(Quantum_fields));
        q_fields_t_star[which_q_field] = (Quantum_fields*)malloc(sizeof(Quantum_fields));
        q_fields_RK1[which_q_field]     = (Quantum_fields *)malloc(sizeof(Quantum_fields));
        q_fields_RK2[which_q_field]     = (Quantum_fields *)malloc(sizeof(Quantum_fields));
        q_fields_RK3[which_q_field]     = (Quantum_fields *)malloc(sizeof(Quantum_fields));
        q_fields_RK4[which_q_field]     = (Quantum_fields *)malloc(sizeof(Quantum_fields));
        q_fields_RK5[which_q_field]     = (Quantum_fields *)malloc(sizeof(Quantum_fields));
        q_fields_RK_sum[which_q_field]  = (Quantum_fields *)malloc(sizeof(Quantum_fields));

        q_fields[which_q_field]->phi       = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields[which_q_field]->chi       = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields[which_q_field]->pi        = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_t_star[which_q_field]->phi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_t_star[which_q_field]->chi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));
        q_fields_t_star[which_q_field]->pi = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        q_fields_RK1[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK1[which_q_field]->chi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK1[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK2[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK2[which_q_field]->chi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK2[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK3[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK3[which_q_field]->chi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK3[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK4[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK4[which_q_field]->chi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK4[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK5[which_q_field]->phi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK5[which_q_field]->chi   = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK5[which_q_field]->pi    = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        q_fields_RK_sum[which_q_field]->phi= (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK_sum[which_q_field]->chi= (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));
        q_fields_RK_sum[which_q_field]->pi = (__complex__ double ***)malloc(number_of_k_modes * sizeof(__complex__ double **));

        alpha_AB_pi_mode[which_q_field] = (__complex__ double***)malloc(number_of_k_modes * sizeof(__complex__ double**));

        for (int k=0; k<number_of_k_modes; k++){

            q_fields[which_q_field]->phi[k]         = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields[which_q_field]->chi[k]         = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields[which_q_field]->pi[k]          = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_t_star[which_q_field]->phi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_t_star[which_q_field]->chi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));
            q_fields_t_star[which_q_field]->pi[k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            q_fields_RK1[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK1[which_q_field]->chi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK1[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK2[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK2[which_q_field]->chi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK2[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK3[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK3[which_q_field]->chi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK3[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK4[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK4[which_q_field]->chi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK4[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK5[which_q_field]->phi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK5[which_q_field]->chi[k]     = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK5[which_q_field]->pi[k]      = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            q_fields_RK_sum[which_q_field]->phi[k]  = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK_sum[which_q_field]->chi[k]  = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));
            q_fields_RK_sum[which_q_field]->pi[k]   = (__complex__ double **)malloc(number_of_l_modes * sizeof(__complex__ double *));

            alpha_AB_pi_mode[which_q_field][k] = (__complex__ double**)malloc(number_of_l_modes * sizeof(__complex__ double*));

            for(int l=0;l<number_of_l_modes;++l){

                q_fields[which_q_field]->phi[k][l]         = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields[which_q_field]->chi[k][l]         = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields[which_q_field]->pi[k][l]          = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_t_star[which_q_field]->phi[k][l] = (__complex__ double*)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_t_star[which_q_field]->chi[k][l] = (__complex__ double*)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_t_star[which_q_field]->pi[k][l] = (__complex__ double*)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_RK1[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK1[which_q_field]->chi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK1[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_RK2[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK2[which_q_field]->chi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK2[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_RK3[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK3[which_q_field]->chi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK3[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_RK4[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK4[which_q_field]->chi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK4[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_RK5[which_q_field]->phi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK5[which_q_field]->chi[k][l]     = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK5[which_q_field]->pi[k][l]      = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                q_fields_RK_sum[which_q_field]->phi[k][l]  = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK_sum[which_q_field]->chi[k][l]  = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));
                q_fields_RK_sum[which_q_field]->pi[k][l]   = (__complex__ double *)malloc(lattice_size_buff * sizeof(__complex__ double));

                alpha_AB_pi_mode[which_q_field][k][l] = (__complex__ double*)malloc(lattice_size_buff * sizeof(__complex__ double));
            }
        }
    }

    double alpha_save[evolve_time_int];
    // double correlator_half[lattice_size][lattice_size];
    // double correlator_half_pp[lattice_size][lattice_size];
    // double correlator_half_cc[lattice_size][lattice_size];
    double cosm_const=0.0;
    save_points();

    double field_save[evolve_time_int_per_five][lattice_size_buff];
    double field_save_q[evolve_time_int_per_five][lattice_size_buff];

    printf("Memory allocation done and took %f\n", omp_get_wtime() - timestart);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////* ACTUAL INITIALISATION AND EVOLUTION *//////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    initial_conditions_quantum      (c_fields, q_fields,   metric);

    // cosm_const=set_cosm_constant    (c_fields, q_fields,   metric);
    cosm_const = 0; /* No backreaction! */

    // printf("The cosmological constant is %.10f,\n", cosm_const);

    initial_conditions_classical    (c_fields, q_fields,   metric);

    set_A_U_lambda_initial          (c_fields,   q_fields,   metric);
    save_stress_tensor(0, cosm_const, c_fields, q_fields, metric);
    save_Aini(metric->A);
    save_stress_tensor_q(0, cosm_const, c_fields, q_fields, metric);
    
    printf("Initial conditions done and took %f\n", omp_get_wtime() -timestart);

    fullEvolution(b_i, a_ij, alpha_AB_pi_mode, c_fields_RK1, c_fields_RK2, c_fields_RK3, c_fields_RK4, c_fields_RK5, c_fields_RK_sum, c_fields,
                 q_fields_RK1, q_fields_RK2, q_fields_RK3, q_fields_RK4, q_fields_RK5, q_fields_RK_sum, q_fields,
                 metric_RK1, metric_RK2, metric_RK3, metric_RK4, metric_RK5, metric_RK_sum, metric, alpha_save, cosm_const, timestart, q_fields_t_star, correlator_half, correlator_half_pp, field_save, field_save_q);
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////* SAVE ALL RELEVANT DATA *///////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    save_correlator_half(20, c_fields, q_fields, correlator_half, correlator_half_pp);
    
    double rho[lattice_size_buff];
    double r[lattice_size_buff];
    make_points(r);
    double ham_r[lattice_size_buff];

    // Calculate stress energy tensor component (t,t)
    for (int i = 0; i < lattice_size_buff; i++) {

        double A, B;
        double phi_phi, chi_chi, pi_pi, chi_pi, del_theta_phi_del_theta_phi_over_r_sq;
        Bi_Linears    bi_linears;


        A = metric->A[i];
        B = metric->B[i];

        set_bi_linears(i, &bi_linears, c_fields, q_fields, metric);

        phi_phi = bi_linears.phi_phi;
        chi_chi = bi_linears.chi_chi;
        pi_pi = bi_linears.pi_pi;
        chi_pi = bi_linears.chi_pi;
        del_theta_phi_del_theta_phi_over_r_sq = bi_linears.del_theta_phi_del_theta_phi_over_r_sq;

        rho[i] = 1.0 / (2.0 * A) * (pi_pi / (B * B) + chi_chi) + 1.0 / B * del_theta_phi_del_theta_phi_over_r_sq;

    } 

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /* Calculate the Ricci scalar */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double ricci_scalar[lattice_size_buff];
    for (int i = 0; i < lattice_size_buff; i++) {
        double A, B, D_B, phi, pi, chi, lambda, U_tilda, D_alpha, K, K_B;
        double D_B_deriv, U_tilda_deriv, lambda_deriv, D_alpha_deriv;

        A             = metric->A[i];
        B             = metric->B[i];
        D_B           = metric->D_B[i];
        D_alpha       = metric->D_alpha[i];
        lambda        = metric->lambda[i];
        U_tilda       = metric->U_tilda[i];
        K             = metric->K[i];
        K_B           = metric->K_B[i];
        phi           = c_fields->phi[i];
        chi           = c_fields->chi[i];
        pi            = c_fields->pi[i];
        D_B_deriv     = first_deriv(i, metric->D_B);    
        U_tilda_deriv = first_deriv(i, metric->U_tilda);
        lambda_deriv  = first_deriv(i, metric->lambda);
        D_alpha_deriv = first_deriv(i, metric->D_alpha);


        ricci_scalar[i] =  D_B_deriv / B - 3.0 * (U_tilda_deriv + 2.0 * D_B_deriv + 4.0 * B * lambda_deriv / A) / A + 8.0 * D_B_deriv / A + 2.0 * K_B * (3.0 * K_B - 2 * K);

    }

    save_field_t(field_save);
    save_field_t_q(field_save_q);
    save_rho(rho);
    // save_mass_r(mass_BH);
    save_ricci(ricci_scalar);
    save_A(metric->A);
    save_B(metric->B);
    save_K(metric->K);
    save_K_B(metric->K_B);
    save_alpha_r(metric->alpha);
    save_D_alpha(metric->D_alpha);
    save_lambda(metric->lambda);
    save_U_tilda(metric->U_tilda);
    save_D_B(metric->D_B);
    

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////* FREE ALL THE MEMORY */////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    free_memory(c_fields,        q_fields,        metric);
    free_memory(c_fields_RK1,    q_fields_RK1,    metric_RK1);
    free_memory(c_fields_RK2,    q_fields_RK2,    metric_RK2);
    free_memory(c_fields_RK3,    q_fields_RK3,    metric_RK3);
    free_memory(c_fields_RK4,    q_fields_RK4,    metric_RK4);
    free_memory(c_fields_RK5,    q_fields_RK5,    metric_RK5);
    free_memory(c_fields_RK_sum, q_fields_RK_sum, metric_RK_sum);

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(q_fields_t_star[which_q_field]->phi[k][l]);
                free(q_fields_t_star[which_q_field]->chi[k][l]);
                free(q_fields_t_star[which_q_field]->pi[k][l]);
            }
            free(q_fields_t_star[which_q_field]->phi[k]);
            free(q_fields_t_star[which_q_field]->chi[k]);
            free(q_fields_t_star[which_q_field]->pi[k]);
        }
        free(q_fields_t_star[which_q_field]->phi);
        free(q_fields_t_star[which_q_field]->chi);
        free(q_fields_t_star[which_q_field]->pi);

        free(q_fields_t_star[which_q_field]);
    }
    free(q_fields_t_star);

    for (int which_q_field = 0; which_q_field < number_of_q_fields; ++which_q_field) {
        for (int k = 0; k < number_of_k_modes; ++k) {
            for (int l = 0; l < number_of_l_modes; ++l) {
                free(alpha_AB_pi_mode[which_q_field][k][l]);
            }
            free(alpha_AB_pi_mode[which_q_field][k]);
        }
        free(alpha_AB_pi_mode[which_q_field]);
    }
    free(alpha_AB_pi_mode);

    for (int i = 0; i < lattice_size; i++) {
        free(correlator_half_pp[i]);
    }
    free(correlator_half_pp);

    for (int i = 0; i < lattice_size; i++) {
        free(correlator_half[i]);
    }
    free(correlator_half);

    // for (int i = 0; i < lattice_size; i++) {
    //     free(correlator_half_cc[i]);
    // }
    // free(correlator_half_cc);

    printf("The code took %f", omp_get_wtime() - timestart);

}