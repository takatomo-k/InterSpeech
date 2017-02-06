#!/usr/bin/perl
use strict;
use warnings;

my $len=0;
my $tmp=0;
while(<>){
	if($_=~/\]/){
		if($tmp>$len){
			$len=$tmp;
			
		}
		$tmp=0;
	}else{
		$tmp++;
	}
}
print "$len\n";
