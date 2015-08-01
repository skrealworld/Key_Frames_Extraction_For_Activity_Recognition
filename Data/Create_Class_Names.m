ClassNames ={'ApplyEyeMakeup','ApplyLipstick','Archery','BabyCrawling','BalanceBeam','BandMarching','BaseballPitch','Basketball','BasketballDunk','BenchPress','Biking','Billiards','BlowDryHair','BlowingCandles','BodyWeightSquats','Bowling','BoxingPunchingBag','BoxingSpeedBag','BreastStroke','BrushingTeeth','CleanAndJerk','CliffDiving','CricketBowling','CricketShot','CuttingInKitchen','Diving','Drumming','Fencing','FieldHockeyPenalty','FloorGymnastics','FrisbeeCatch','FrontCrawl','GolfSwing','Haircut','Hammering','HammerThrow','HandstandPushups','HandstandWalking','HeadMassage','HighJump','HorseRace','HorseRiding','HulaHoop','IceDancing','JavelinThrow','JugglingBalls','JumpingJack','JumpRope','Kayaking','Knitting','LongJump','Lunges','MilitaryParade','Mixing','MoppingFloor','Nunchucks','ParallelBars','PizzaTossing','PlayingCello','PlayingDaf','PlayingDhol','PlayingFlute','PlayingGuitar','PlayingPiano','PlayingSitar','PlayingTabla','PlayingViolin','PoleVault','PommelHorse','PullUps','Punch','PushUps','Rafting','RockClimbingIndoor','RopeClimbing','Rowing','SalsaSpin','ShavingBeard','Shotput','SkateBoarding','Skiing','Skijet','SkyDiving','SoccerJuggling','SoccerPenalty','StillRings','SumoWrestling','Surfing','Swing','TableTennisShot','TaiChi','TennisSwing','ThrowDiscus','TrampolineJumping','Typing','UnevenBars','VolleyballSpiking','WalkingWithDog','WallPushups','WritingOnBoard','YoYo'};
for n1= 1:101
cd (strcat('~/skdata/UCF-101/',ClassNames{n1}));
%cd ('/Users/SK_Mac/Documents/MATLAB/UCF-101/BalanceBeam');
files = dir('*.ogv');
% Loop through each
for id = 1:length(files)
  movefile(files(id).name, strcat(ClassNames{n1},'(',num2str(id),').ogv' ));
end
end
%for f1=1:101
%cd (strcat('~/Documents/MATLAB/UCF-101/',ClassNames{f1}));
%d= dir; 
%file_numbers(f1) = length(d);
%end
