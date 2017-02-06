#!/usr/bin/perl
use strict;
use warnings;
use utf8;
use File::Basename 'basename', 'dirname';

if(@ARGV!=2){
	print "This script reoder text for mfcc file.";
	print "usage: check_oder.pl SrcFile.mfcc TrgFile.text\n";
}elsif($ARGV[0]=~/\.txt/){
	print "src file should be mfcc file\n";
}
open(SRC,$ARGV[0]);
open(TRG,$ARGV[1]);

my @trg=<TRG>;
#check oder
my $i=0;
while(<SRC>){
	if($_=~/\[/){
		my @tmp=split(/ /,$_);
		my @trg_name=split(/ /,$trg[$i]);
		
		if($tmp[0] ne $trg_name[0]){
			print "Text file's oder not much to mfcc file [$tmp[0]|$trg_name[0]] \n";
			print "Start reoder text file.\n";
			$i=-1;
			last;
		}else{
			$i++;
		}
	}
}
close(SRC);
close(TRG);

if($i<0){
open(MFCC,$ARGV[0]);
open(TXT,">",$ARGV[1]);
	#make trg hash
	my @out;
	my %trg_hash;
	foreach my $line(@trg){
	      	my @tmp=split(/ /,$line);
		$trg_hash{$tmp[0]}=$line;
	}
 	#debug
	#print "keys(%trg_hash)\n";
	while(<MFCC>){
		if($_=~/\[/){
			my @tmp=split(/ /,$_);
			if(exists($trg_hash{$tmp[0]})){
				push(@out,$trg_hash{$tmp[0]})
			}else{
				print "$tmp[0] not exists in text file.\n";	
				exit;
			}
		}
	}
	
	print TXT @out;	
	print "Reoder Done [input $#trg+1 lines | output $#out+1 lines]\n"
}else{
	print "Oder is correct.\n";
}


