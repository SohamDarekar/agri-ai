import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { MdEco, MdBugReport, MdLightMode, MdDarkMode, MdTrendingUp, MdScience, MdLocalFlorist, MdAttachMoney } from 'react-icons/md';
import { GiWheat, GiPlantSeed } from 'react-icons/gi';
import { FeatureCard, LanguageSwitcher, AgriculturalNews, Card } from '../components';
import { useAppSelector } from '../store';
import { useTheme } from '../hooks';
import { useLanguage } from '../contexts/LanguageContext';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const isOnline = useAppSelector(state => state.app.isOnline);
  const { theme, toggleTheme } = useTheme();
  const { t } = useLanguage();

  const features = [
    {
      title: t('cropRecommendationTitle'),
      description: t('cropRecommendationDesc'),
      icon: <GiPlantSeed />,
      path: '/crop-recommendation',
      requiresInternet: true,
      badge: 'Smart AI'
    },
    {
      title: t('yieldPredictionTitle'),
      description: t('yieldPredictionDesc'),
      icon: <GiWheat />,
      path: '/yield-prediction',
      requiresInternet: true,
      badge: 'ML Powered'
    },
    {
      title: t('diseaseDetectionTitle'),
      description: t('diseaseDetectionDesc'),
      icon: <MdBugReport />,
      path: '/disease-detection',
      requiresInternet: false,
      badge: 'Offline Ready'
    },
    {
      title: t('marketInsights'),
      description: t('marketInsightsDesc'),
      icon: <MdTrendingUp />,
      path: '/market-insights',
      requiresInternet: true,
      badge: 'Real-time Data'
    },
    {
      title: t('profitCalculatorTitle'),
      description: t('profitCalculatorDesc'),
      icon: <MdAttachMoney />,
      path: '/profit-calculator',
      requiresInternet: true,
      badge: 'Smart Analytics'
    }
  ];

  const stats = [
    { icon: <MdTrendingUp />, value: '95%', label: t('accuracyRate') },
    { icon: <MdScience />, value: '22+', label: t('cropsSupported') },
    { icon: <MdLocalFlorist />, value: '38+', label: t('diseasesDetected') }
  ];

  return (
    <div className="min-h-screen-safe">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-6 sm:py-8 mb-6 sm:mb-8"
      >
        <div className="flex items-center justify-between mb-6 px-2">
          <div className="flex-1" />
          <div className="flex items-center space-x-2 sm:space-x-3">
            <MdEco className="text-3xl sm:text-4xl text-primary-500" />
            <h1 className="text-2xl sm:text-3xl font-bold text-gradient">
              Krishi Sahayak
            </h1>
          </div>
          <div className="flex-1 flex justify-end items-center gap-2">
            <LanguageSwitcher />
            <button
              onClick={toggleTheme}
              className="touch-target p-2 rounded-lg bg-white dark:bg-gray-800 shadow-soft hover:shadow-card transition-all duration-200 text-gray-600 dark:text-gray-400 hover:text-primary-500"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? <MdDarkMode size={20} /> : <MdLightMode size={20} />}
            </button>
          </div>
        </div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="px-4"
        >
          <h2 className="text-3xl xs:text-4xl sm:text-5xl md:text-6xl font-bold text-gray-900 dark:text-gray-100 mb-4 leading-tight">
            {t('welcomeTitle')}
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-base sm:text-lg md:text-xl max-w-3xl mx-auto leading-relaxed mb-6 sm:mb-8">
            {t('welcomeSubtitle')}
          </p>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-3 gap-6 sm:gap-8 max-w-lg sm:max-w-2xl mx-auto mb-8 sm:mb-12 px-4"
        >
          {stats.map((stat, index) => (
            <motion.div 
              key={index} 
              className="text-center group"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + index * 0.1, duration: 0.4 }}
            >
              <div className="flex justify-center mb-3">
                <div className="p-3 sm:p-4 bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/30 dark:to-primary-800/30 rounded-2xl group-hover:scale-110 transition-transform duration-300">
                  <div className="text-xl sm:text-2xl text-primary-600 dark:text-primary-400">
                    {stat.icon}
                  </div>
                </div>
              </div>
              <div className="text-xl sm:text-3xl font-bold text-gray-900 dark:text-gray-100 mb-1">
                {stat.value}
              </div>
              <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 font-medium">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </motion.header>

      {/* Feature Cards */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mb-12 sm:mb-16 px-4"
      >
        <div className="text-center mb-8 sm:mb-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="max-w-3xl mx-auto"
          >
            <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
              {t('features')}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-base sm:text-lg leading-relaxed">
              {t('exploreTools')}
            </p>
          </motion.div>
        </div>
        
        <div className="max-w-7xl mx-auto">
          <div className="mobile-feature-grid">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  delay: 0.5 + index * 0.15,
                  duration: 0.6,
                  ease: "easeOut"
                }}
                className="h-full"
              >
                <FeatureCard
                  title={feature.title}
                  description={feature.description}
                  icon={feature.icon}
                  onClick={() => navigate(feature.path)}
                  disabled={feature.requiresInternet && !isOnline}
                  badge={feature.badge}
                />
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* News Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="max-w-4xl mx-auto mb-8 sm:mb-12 px-4"
      >
        <AgriculturalNews />
      </motion.div>

      {/* Additional Info Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.0 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8 max-w-5xl mx-auto mb-16 px-4"
      >
        {/* Offline Capability */}
        <motion.div
          whileHover={{ y: -4, scale: 1.02 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="p-6 sm:p-8 bg-gradient-to-br from-green-50 via-green-50 to-green-100 dark:from-green-900/30 dark:via-green-900/20 dark:to-green-800/30 border-green-200/50 dark:border-green-700/50 hover:border-green-300 dark:hover:border-green-600 transition-all duration-300">
            <div className="flex items-start mb-4">
              <div className="p-3 bg-gradient-to-br from-green-500 to-green-600 text-white rounded-xl mr-4 shadow-lg">
                <MdScience className="text-xl" />
              </div>
              <div className="flex-1">
                <h4 className="text-lg font-bold text-green-900 dark:text-green-100 mb-2">
                  {t('offlineAiModels')}
                </h4>
                <p className="text-sm sm:text-base text-green-800 dark:text-green-200 leading-relaxed">
                  {t('offlineAiModelsDesc')}
                </p>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Data Privacy */}
        <motion.div
          whileHover={{ y: -4, scale: 1.02 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="p-6 sm:p-8 bg-gradient-to-br from-blue-50 via-blue-50 to-blue-100 dark:from-blue-900/30 dark:via-blue-900/20 dark:to-blue-800/30 border-blue-200/50 dark:border-blue-700/50 hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-300">
            <div className="flex items-start mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-xl mr-4 shadow-lg">
                <MdLocalFlorist className="text-xl" />
              </div>
              <div className="flex-1">
                <h4 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-2">
                  {t('privacyFirst')}
                </h4>
                <p className="text-sm sm:text-base text-blue-800 dark:text-blue-200 leading-relaxed">
                  {t('privacyFirstDesc')}
                </p>
              </div>
            </div>
          </Card>
        </motion.div>
      </motion.div>

      {/* Online Status Message */}
      {!isOnline && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto mb-8 px-4"
        >
          <div className="p-4 sm:p-6 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/30 dark:to-orange-900/30 border border-amber-200/60 dark:border-amber-700/60 rounded-2xl text-center shadow-lg">
            <div className="flex items-center justify-center mb-2">
              <div className="text-2xl mr-2">📶</div>
              <p className="text-amber-800 dark:text-amber-200 font-semibold text-sm sm:text-base">
                {t('offlineMessage')}
              </p>
            </div>
            <p className="text-amber-700 dark:text-amber-300 text-xs sm:text-sm opacity-80">
              Some features may be limited while offline
            </p>
          </div>
        </motion.div>
      )}

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="text-center mt-16 py-8 border-t border-gray-200 dark:border-gray-700"
      >
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-center mb-4">
            <MdEco className="text-2xl text-primary-500 mr-2" />
            <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Krishi Sahayak
            </span>
          </div>
          <p className="text-gray-500 dark:text-gray-400 text-sm mb-2">
            {t('empoweringFarmers')}
          </p>
          <p className="text-gray-400 dark:text-gray-500 text-xs">
            {t('builtFor')}
          </p>
        </div>
      </motion.footer>
    </div>
  );
};

export default HomePage;