#!/usr/bin/perl
use strict;
use warnings;
use utf8;
use File::Basename 'basename', 'dirname';

if(@ARGV!=2){
	print "usage: ark2txt.pl InputDir OutputDir\n";
}

my $indir=$ARGV[0];
my $outdir=$ARGV[1];

while(<$indir/*>){
	if($_=~/\.ark/){
		my $mfcc= basename $_;
		$mfcc=~s/\.ark/\.mfcc/;
		system("copy-feats ark:$_ ark,t:$outdir/$mfcc");

	}
}

