module.exports = {
    webpack: {
      configure: (webpackConfig) => {

        const sourceMapLoaderRule = webpackConfig.module.rules.find(
          (rule) =>
            rule.loader &&
            rule.loader.includes("source-map-loader")
        );
        if (sourceMapLoaderRule) {
          sourceMapLoaderRule.exclude = /node_modules\/plotly.js/;
        }
        return webpackConfig;
      },
    },
  };
  