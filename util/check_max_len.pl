#!/usr/bin/perl
use strict;
use warnings;
my $len=0;
while(<>){
	my @tmp=split(/ /,$_);
	if($#tmp>$len){
		$len=$#tmp;
	}
}

print "$len\n";
