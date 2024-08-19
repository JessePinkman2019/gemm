%
% Clear all variables and close all graphs
%

clear all
close all

%
% Get max_gflops from /proc/cpuinfo by reading the parameters
% set in file proc_parameters.m
%

run('../header/proc_parameters')

max_gflops = nflops_per_cycle * nprocessors * GHz_of_processor;

run('../output/output_MMult0')

version_old = version;

plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'bo-.;MMult0;' );
last = size( MY_MMult, 1 );

hold on

axis( [ 0 MY_MMult( last,1 ) 0 max_gflops ] );

xlabel( 'm = n = k' );
ylabel( 'GFLOPS/sec.' );

%
% Read in second data set and plot it.
%

run('../output/output_MMult1')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'r-*;MMult1;' );


run('../output/output_MMult2')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'g-*;MMult2;' );


run('../output/output_MMult3')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'y-*;MMult3;' );

run('../output/output_MMult4')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'k-*;MMult4;' );

run('../output/output_MMult5')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'm-*;MMult5;' );

run('../output/output_MMult6')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'b-*;MMult6;' );

run('../output/output_MMult7')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'g-*;MMult7;' );

run('../output/output_MMult8')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'r-*;MMult8;' );

run('../output/output_MMult18')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'k-*;MMult18;' );
run('../output/output_MMult19')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'm-*;MMult19;' );
run('../output/output_MMult20')


plot( MY_MMult( :,1 ), MY_MMult( :,2 ), 'y-*;MMult20;' );


filename = sprintf( "compare_all" );
print( filename, '-dpng' );
